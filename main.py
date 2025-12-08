# /main.py

import os
import torch
import json
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
    # Configurar logger principal com arquivo de log
    logger = setup_logger(
        name='PINN_Main',
        log_dir=PATHS.get('results_dir', 'resultados'),
        level=20  # INFO
    ) 
    logger.info("Iniciando Pipeline Heston-LSTM Híbrido")
    
    # --- 1. Preparação dos Dados ---
    logger.info("[Fase 1] Carregando dados e gerando sequências temporais...")
    df_juros = carregar_taxa_juros(PATHS['selic_data'])
    if df_juros is None:
        logger.error("Falha ao carregar taxa de juros.")
        return

    # O novo loader faz todo o trabalho pesado: sliding windows + normalização
    full_dataset, data_stats = criar_dataset_hibrido(
        caminho_pasta_opcoes=PATHS['raw_data'],
        df_juros=df_juros,
        seq_length=DATA_CONFIG['sequence_length']
    )
    
    if full_dataset is None:
        logger.error("Nenhum dado encontrado ou processado.")
        return
    
    # ===== CORREÇÃO 5: EXTRAIR NÚMERO REAL DE ATIVOS =====
    num_assets_detected = len(data_stats.get('asset_map', {}))
    logger.info(f"Ativos detectados no dataset: {num_assets_detected}")
    
    if num_assets_detected > 0:
        # Atualizar MODEL_CONFIG dinamicamente
        MODEL_CONFIG['num_assets'] = max(num_assets_detected, 10)  # Mínimo de segurança
        logger.info(f"MODEL_CONFIG['num_assets'] atualizado para: {MODEL_CONFIG['num_assets']}")
        
        # Log dos ativos encontrados
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

    # Divisão Treino / Validação
    total_size = len(full_dataset)
    val_size = int(total_size * DATA_CONFIG['test_size'])
    train_size = total_size - val_size
    
    # Generator com seed fixa para reprodutibilidade
    generator = torch.Generator().manual_seed(DATA_CONFIG['random_state'])
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Otimização de DataLoader: num_workers > 0 usa multiprocessamento para carregar dados
    # pin_memory=True acelera transferência para GPU
    train_loader = DataLoader(train_dataset, 
                              batch_size=TRAINING_CONFIG['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=TRAINING_CONFIG['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)
    
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
    
    # Executa o treino e salva o histórico
    trainer.train()
    
    # --- 4. Fine-Tuning Automático ---
    logger.info("[Fase 4] Iniciando Especialização por Ativo...")
    
    # Recarrega o melhor modelo base
    model.load_state_dict(torch.load(os.path.join(PATHS['model_save_dir'], 'best_model_weights.pth')))
    # Instancia o Tuner usando o dataset completo que já está na memória (eficiência!)
    tuner = FineTuner(model, full_dataset, data_stats, TRAINING_CONFIG, PATHS)
    # Roda para todos
    tuner.fine_tune_all(epochs=5, 
                        lr=1e-5
                    )

    # --- 5. Visualização e Diagnóstico ---
    logger.info("[Fase 5] Gerando Diagnósticos Visuais...")
    
    # O trainer salva o histórico em results_dir/training_history.csv, vamos garantir que usamos esse caminho
    history_file_path = os.path.join(PATHS['results_dir'], 'training_history.csv')
    
    # Verifica se o histórico foi criado antes de tentar plotar
    if not os.path.exists(history_file_path):
        logger.warning(f"Arquivo de histórico não encontrado em {history_file_path}. Plots de Loss serão ignorados.")
    
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