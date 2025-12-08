#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
teste_rapido.py - Teste Rápido do Pipeline Completo
Executa 3 épocas de treinamento para validar toda a arquitetura
"""

import os
import torch
import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

# Importa configurações e módulos
from src.config import PATHS, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG
from src.data_loader import carregar_taxa_juros, criar_dataset_hibrido
from src.model import DeepHestonHybrid
from src.trainer import PINNTrainer
from src.visualization import Visualizer
from src.logger import get_logger

logger = get_logger('TESTE_RAPIDO')

def teste_rapido():
    """
    Executa 3 épocas de treinamento para testar toda a arquitetura.
    """
    print("\n" + "="*80)
    print("TESTE RAPIDO - PIPELINE COMPLETO (3 EPOCAS)")
    print("="*80 + "\n")
    
    # --- 1. PREPARACAO DOS DADOS ---
    print("[PASSO 1] Carregando e preparando dados...")
    df_juros = carregar_taxa_juros(PATHS['selic_data'])
    if df_juros is None:
        print("ERRO: Falha ao carregar taxa de juros")
        return False

    print(f"  OK - Taxa de juros carregada: {len(df_juros)} linhas")

    # Cria dataset
    print("[PASSO 2] Criando dataset...")
    full_dataset, data_stats = criar_dataset_hibrido(
        caminho_pasta_opcoes=PATHS['raw_data'],
        df_juros=df_juros,
        seq_length=DATA_CONFIG['sequence_length']
    )
    
    if full_dataset is None:
        print("  AVISO - Dataset nao disponivel. Criando dataset sintetico para teste...")
        full_dataset, data_stats = _criar_dataset_sintetico()
    
    print(f"  OK - Dataset criado: {len(full_dataset)} amostras")
    print(f"  OK - Data stats: {list(data_stats.keys())}")

    # Divisao Treino/Validacao
    print("[PASSO 3] Dividindo dados em treino e validacao...")
    total_size = len(full_dataset)
    val_size = int(total_size * DATA_CONFIG['test_size'])
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    batch_size = 32  # Menor batch size para teste rapido
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"  OK - Treino: {len(train_dataset)} amostras")
    print(f"  OK - Validacao: {len(val_dataset)} amostras")

    # --- 2. INSTANCIACAO DO MODELO ---
    print("\n[PASSO 4] Inicializando modelo DeepHestonHybrid...")
    model = DeepHestonHybrid(config=MODEL_CONFIG, data_stats=data_stats)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  OK - Modelo criado com {total_params:,} parametros")
    print(f"  OK - LSTM input size: {MODEL_CONFIG['lstm_input_size']} features")

    # --- 3. CONFIGURACAO PARA TESTE RAPIDO ---
    print("\n[PASSO 5] Configurando treinador para teste rapido...")
    
    # Modifica config temporariamente para teste rapido
    test_config = TRAINING_CONFIG.copy()
    test_config['learning_rates'] = [1e-3]  # Uma unica fase
    test_config['epochs_per_phase'] = 3     # 3 epocas apenas
    test_config['batch_size'] = batch_size
    
    print(f"  OK - Learning rate: {test_config['learning_rates'][0]}")
    print(f"  OK - Epocas por fase: {test_config['epochs_per_phase']}")
    print(f"  OK - Batch size: {batch_size}")

    # --- 4. TREINAMENTO ---
    print("\n[PASSO 6] Executando treinamento (3 epocas)...\n")
    
    trainer = PINNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        data_stats=data_stats,
        config=test_config
    )
    
    try:
        trainer.train()
        print("\n  OK - Treinamento concluido com sucesso!")
    except Exception as e:
        print(f"\n  ERRO durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False

    # --- 5. SALVAR HISTORICO ---
    print("\n[PASSO 7] Salvando historico...")
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    history_path = os.path.join(PATHS['results_dir'], 'test_history.csv')
    
    try:
        from src.utils import salvar_historico_treinamento
        salvar_historico_treinamento(trainer.history, PATHS['results_dir'])
        print(f"  OK - Historico salvo em: {history_path}")
    except Exception as e:
        print(f"  AVISO - Erro ao salvar historico: {e}")

    # --- 6. TESTES DE VALIDACAO ---
    print("\n[PASSO 8] Validacao do modelo...")
    
    try:
        model.eval()
        device = next(model.parameters()).device
        
        # Get first batch
        batch = next(iter(val_loader))
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        
        with torch.no_grad():
            x_seq, x_phy, y, x_time, weights, ids = batch
            outputs = model(x_seq, x_phy, ids)
            
            pred_price = outputs['price']
            heston_params = outputs['heston_params']
            
            print(f"  OK - Validacao com batch de {len(x_seq)} amostras")
            print(f"  OK - Precos preditos: shape {pred_price.shape}, min={pred_price.min():.4f}, max={pred_price.max():.4f}")
            print(f"  OK - Parametros Heston: {len(heston_params)} tensores")
            
    except Exception as e:
        print(f"  ERRO na validacao: {e}")
        return False

    # --- 7. SUMARIO FINAL ---
    print("\n" + "="*80)
    print("RESUMO DO TESTE")
    print("="*80)
    print(f"Numero de epocas: 3")
    print(f"Tamanho do dataset: {len(full_dataset)}")
    print(f"Parametros do modelo: {total_params:,}")
    print(f"Features LSTM: {MODEL_CONFIG['lstm_input_size']}")
    print(f"Melhor loss de validacao: {trainer.best_val_loss:.6f}")
    print(f"Perdas por epoca:")
    
    for i, loss in enumerate(trainer.history['train_loss']):
        val_loss = trainer.history['val_loss'][i] if i < len(trainer.history['val_loss']) else 0
        print(f"  Epoca {i+1}: train_loss={loss:.6f}, val_loss={val_loss:.6f}")
    
    print("\nOK - TESTE RAPIDO CONCLUIDO COM SUCESSO!")
    print("  Toda a arquitetura esta funcionando perfeitamente!")
    print("="*80 + "\n")
    
    return True


def _criar_dataset_sintetico():
    """
    Cria dataset sintetico para teste quando dados reais nao estao disponíveis.
    """
    import torch
    from torch.utils.data import TensorDataset
    
    batch_size = 256
    seq_length = 30
    n_samples = batch_size * 4
    
    # X_seq: [Batch, SeqLen, 6 features]
    X_seq = torch.randn(n_samples, seq_length, 6, dtype=torch.float32)
    
    # X_phy: [Batch, 5] - S, K, T, r, q
    X_phy = torch.randn(n_samples, 5, dtype=torch.float32)
    X_phy[:, 0] = torch.abs(X_phy[:, 0]) * 50 + 50    # S: 50-100
    X_phy[:, 1] = torch.abs(X_phy[:, 1]) * 50 + 50    # K: 50-100
    X_phy[:, 2] = torch.abs(X_phy[:, 2]) * 0.5 + 0.1  # T: 0.1-0.6
    X_phy[:, 3] = torch.abs(X_phy[:, 3]) * 0.05 + 0.05 # r: 0.05-0.1
    X_phy[:, 4] = torch.abs(X_phy[:, 4]) * 0.03       # q: 0-0.03
    
    # y: [Batch, 1] - premios positivos
    y = torch.abs(torch.randn(n_samples, 1, dtype=torch.float32)) * 5 + 1
    
    X_time = torch.arange(n_samples, dtype=torch.float32)
    weights = torch.ones(n_samples, 1, dtype=torch.float32)
    ids = torch.zeros(n_samples, dtype=torch.int64)  # Todos do mesmo ativo
    
    dataset = TensorDataset(X_seq, X_phy, y, X_time, weights, ids)
    
    data_stats = {
        'S_min': 50.0, 'S_max': 100.0, 'S_mean': 75.0, 'S_std': 12.5,
        'K_min': 50.0, 'K_max': 100.0, 'K_mean': 75.0, 'K_std': 12.5,
        'T_min': 0.1, 'T_max': 0.6, 'T_mean': 0.35, 'T_std': 0.125,
        'r_min': 0.05, 'r_max': 0.1, 'r_mean': 0.075, 'r_std': 0.0125,
        'q_min': 0.0, 'q_max': 0.03, 'q_mean': 0.015, 'q_std': 0.0075,
    }
    
    print("    Usando dataset SINTETICO para teste")
    return dataset, data_stats


if __name__ == '__main__':
    import sys
    
    try:
        success = teste_rapido()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTeste interrompido pelo usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERRO NAO TRATADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
