# run_single_asset_experiment.py
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Imports do projeto
from src.config import PATHS, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG
from src.data_loader import carregar_taxa_juros, criar_dataset_hibrido
from src.model import DeepHestonHybrid
from src.trainer import PINNTrainer
from src.logger import setup_logger
from src.visualization import Visualizer

def run_single_asset(target_asset="PETR4"): # Escolha o ticker base da opção (ex: PETR4)
    
    logger = setup_logger(name=f'Single_{target_asset}', log_dir='resultados')
    logger.info(f"=== INICIANDO EXPERIMENTO CIENTÍFICO: SINGLE ASSET ({target_asset}) ===")
    logger.info("Objetivo: Validar se o viés negativo desaparece ao isolar um único regime de volatilidade.")

    # 1. Carrega Juros
    df_juros = carregar_taxa_juros(PATHS['selic_data'])
    
    # 2. Carrega Dataset Completo
    logger.info("Carregando dataset bruto...")
    # Nota: O data_loader original carrega tudo. Vamos filtrar DEPOIS de carregar
    # para não precisar reescrever o loader complexo agora.
    full_dataset, data_stats = criar_dataset_hibrido(
        PATHS['raw_data'], df_juros, DATA_CONFIG['sequence_length']
    )
    
    # 3. FILTRAGEM MANUAL PARA SINGLE ASSET
    logger.info(f"Filtrando apenas ativo {target_asset}...")
    
    # Recupera o ID do ativo alvo
    asset_map = data_stats['asset_map']
    # Procura chaves no asset_map que contenham o target (ex: 'PETR' in 'PETR4')
    target_ids = [v for k, v in asset_map.items() if target_asset in k]
    
    if not target_ids:
        logger.error(f"Ativo {target_asset} não encontrado no asset_map!")
        return

    # O asset_id é o 6º elemento do TensorDataset (índice 5)
    # Estrutura: x_seq, x_phy, y, x_time, weights, asset_ids
    all_ids = full_dataset.tensors[5].numpy()
    
    # Máscara booleana
    mask = np.isin(all_ids, target_ids)
    indices = np.where(mask)[0]
    
    logger.info(f"Total amostras: {len(all_ids)}. Amostras {target_asset}: {len(indices)}")
    
    # Cria Subset
    subset = torch.utils.data.Subset(full_dataset, indices)
    
    # Divide Treino/Validação (80/20) apenas desse ativo
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(subset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    # 4. Ajuste na Configuração do Modelo
    # Importante: O modelo ainda vai ter 'num_assets' original, mas só usaremos os IDs filtrados.
    # Isso não afeta o teste.
    
    model = DeepHestonHybrid(MODEL_CONFIG, data_stats)
    
    # 5. Treinamento
    # Ajuste de Config para o teste (pode ser mais curto)
    TEST_CONFIG = TRAINING_CONFIG.copy()
    TEST_CONFIG['epochs_per_phase'] = 150 # Mais rápido para validar
    
    trainer = PINNTrainer(model, train_loader, val_loader, data_stats, TEST_CONFIG)
    trainer.train()
    
    # 6. Diagnóstico Rápido
    logger.info("Gerando visualização de validação...")
    viz = Visualizer(model, os.path.join(PATHS['results_dir'], 'training_history.csv'), 
                     val_loader, data_stats, VIZ_CONFIG)
    
    # Força plot de scatter para vermos o viés
    viz.plot_prediction_scatter()
    
    logger.info("Experimento Finalizado.")

if __name__ == "__main__":
    # Escolha um ativo que tenha MUITOS dados na sua base (ex: PETR4, VALE3 ou BOVA11)
    # Olhe seus CSVs para confirmar o prefixo
    run_single_asset("PETR")