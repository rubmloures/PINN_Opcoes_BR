# /main.py

import os
import sys
import torch
import json
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

# Importa as configurações e módulos
from src.config import PATHS, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG
from src.data_loader import carregar_taxa_juros, criar_dataset_hibrido
from src.model import DeepHestonHybrid
from src.trainer import PINNTrainer
from src.visualization import Visualizer
from src.logger import setup_logger
from src.fine_tune import FineTuner


def run_pipeline():
    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(description="Pipeline PINN Heston-LSTM Híbrido")
    parser.add_argument(
        '--optimize', action='store_true',
        help='Executa otimização Optuna antes do treino principal'
    )
    parser.add_argument(
        '--n-trials', type=int, default=50,
        help='Número de trials Optuna (padrão: 50, requer --optimize)'
    )
    parser.add_argument(
        '--epochs-per-trial', type=int, default=5,
        help='Épocas por trial Optuna (padrão: 5, requer --optimize)'
    )
    args, _ = parser.parse_known_args()

    # Configurar logger principal
    logger = setup_logger(
        name='PINN_Main',
        log_dir=PATHS.get('results_dir', 'resultados'),
        level=20  # INFO
    )
    logger.info("Iniciando Pipeline Heston-LSTM Híbrido")

    # --- 0. Otimização Optuna (opcional) ---
    if args.optimize:
        logger.info(
            f"[Fase 0] Otimização Optuna ({args.n_trials} trials, "
            f"{args.epochs_per_trial} épocas/trial)..."
        )
        try:
            from src.optuna_optimizer import PINNOptunaOptimizer
            opt = PINNOptunaOptimizer(
                n_trials=args.n_trials,
                epochs_per_trial=args.epochs_per_trial,
            )
            study = opt.run()
            best = study.best_params
            logger.info(f"Melhores parâmetros Optuna: {best}")
            # Atualiza MODEL_CONFIG
            for k in ['lstm_hidden_size', 'lstm_layers', 'lstm_dropout',
                      'pinn_hidden_layers', 'pinn_neurons', 'fourier_features',
                      'activation', 'asset_embedding_dim']:
                if k in best:
                    MODEL_CONFIG[k] = best[k]
            # Atualiza TRAINING_CONFIG
            for k in ['batch_size', 'weight_data', 'weight_pde',
                      'warmup_epochs', 'rampup_epochs',
                      'lambda_bc', 'lambda_reg', 'lambda_feller']:
                if k in best:
                    TRAINING_CONFIG[k] = best[k]
            logger.info("Configs atualizadas com parâmetros Optuna.")
        except Exception as e:
            logger.error(f"Otimização Optuna falhou: {e}. Continuando com configs padrão.")

    # --- 1. Preparação dos Dados ---
    logger.info("[Fase 1] Carregando dados e gerando sequências temporais...")
    df_juros = carregar_taxa_juros(PATHS['selic_data'])
    if df_juros is None:
        logger.error("Falha ao carregar taxa de juros.")
        return

    full_dataset, data_stats = criar_dataset_hibrido(
        caminho_pasta_opcoes=PATHS['raw_data'],
        df_juros=df_juros,
        seq_length=DATA_CONFIG['sequence_length']
    )

    if full_dataset is None:
        logger.error("Nenhum dado encontrado ou processado.")
        return

    # Extrai número real de ativos
    num_assets_detected = len(data_stats.get('asset_map', {}))
    logger.info(f"Ativos detectados no dataset: {num_assets_detected}")

    if num_assets_detected > 0:
        MODEL_CONFIG['num_assets'] = max(num_assets_detected, 10)
        logger.info(f"MODEL_CONFIG['num_assets'] atualizado para: {MODEL_CONFIG['num_assets']}")
        if 'asset_names' in data_stats:
            logger.info(f"Ativos: {', '.join(data_stats['asset_names'])}")
    else:
        logger.warning("Nenhum ativo detectado! Usando configuração padrão.")

    # Salva as estatísticas para inferência futura
    os.makedirs(PATHS['model_save_dir'], exist_ok=True)
    stats_path = os.path.join(PATHS['model_save_dir'], 'data_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(data_stats, f, indent=4)
    logger.info(f"Estatísticas salvas em: {stats_path}")

    # --- Divisão Temporal ---
    # Treino: Até 31/12/2024 | Validação: 01/01/2025 a 30/06/2025
    cutoff_ts = pd.Timestamp('2024-12-31').timestamp()
    
    # O dataset hibrido retorna (x_seq, x_phy, y, timestamps, weights, asset_ids)
    # Pegamos o tensor de timestamps (index 3)
    all_timestamps = full_dataset.tensors[3].flatten().numpy()
    
    train_indices = np.where(all_timestamps <= cutoff_ts)[0]
    val_indices = np.where(all_timestamps > cutoff_ts)[0]
    
    if len(val_indices) == 0:
        logger.warning("AVISO: Nenhum dado encontrado para o período de validação (2025).")
        logger.warning("Verifique se os CSVs de 2025 estão na pasta 'dados/brutos'.")
        # Fallback para split aleatório se 2025 estiver vazio, para não quebrar o pipeline
        total_size = len(full_dataset)
        val_size = int(total_size * DATA_CONFIG['test_size'])
        train_size = total_size - val_size
        generator = torch.Generator().manual_seed(DATA_CONFIG['random_state'])
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    else:
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        logger.info(f"Divisão Temporal Aplicada:")
        logger.info(f"  > Treino: Até 31/12/2024 ({len(train_dataset)} amostras)")
        logger.info(f"  > Validação: 2025 Q1/Q2 ({len(val_dataset)} amostras)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )

    logger.info(f"Datasets prontos: Treino ({len(train_dataset)}) | Validação ({len(val_dataset)})")

    # --- 2. Instanciação do Modelo ---
    logger.info("[Fase 2] Inicializando DeepHestonHybrid...")
    model = DeepHestonHybrid(config=MODEL_CONFIG, data_stats=data_stats)

    # --- 3. Treinamento ---
    logger.info("[Fase 3] Iniciando Treinamento...")
    trainer = PINNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        data_stats=data_stats,
        config=TRAINING_CONFIG
    )

    # Executa o treino (inclui callback de normalização pré-treino)
    trainer.train()

    # --- 4. Fine-Tuning Automático ---
    logger.info("[Fase 4] Iniciando Especialização por Ativo...")
    model.load_state_dict(
        torch.load(os.path.join(PATHS['model_save_dir'], 'best_model_weights.pth'))
    )
    tuner = FineTuner(model, full_dataset, data_stats, TRAINING_CONFIG, PATHS)
    tuner.fine_tune_all(epochs=5, lr=1e-5)

    # --- 5. Visualização e Diagnóstico ---
    logger.info("[Fase 5] Gerando Diagnósticos Visuais...")

    history_file_path = os.path.join(PATHS['results_dir'], 'training_history.csv')
    if not os.path.exists(history_file_path):
        logger.warning(
            f"Arquivo de histórico não encontrado em {history_file_path}. "
            "Plots de Loss serão ignorados."
        )

    viz = Visualizer(
        model=model,
        history_path=history_file_path,
        val_loader=val_loader,
        data_stats=data_stats,
        config=VIZ_CONFIG
    )

    viz.plot_all()

    logger.info("Pipeline Finalizado com Sucesso")
    logger.info(f"Modelo salvo em: {PATHS['model_save_dir']}")
    logger.info(f"Plots salvos em: {PATHS['plot_save_dir']}")


if __name__ == '__main__':
    run_pipeline()