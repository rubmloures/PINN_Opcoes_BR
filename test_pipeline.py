#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_pipeline.py - Script de Teste de Integridade e QA.
Este script valida a escala das entradas, o funcionamento dos módulos (Loader, Model, Trainer)
e executa um treinamento curto para garantir que o pipeline e os gráficos estão funcionais.
"""

import os
import sys
import torch
import json
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

# Configuração de Path
sys.path.append(os.getcwd())

# Importações do Projeto
from src.config import PATHS, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG
from src.data_loader import carregar_taxa_juros, criar_dataset_hibrido
from src.model import DeepHestonHybrid
from src.trainer import PINNTrainer
from src.visualization import Visualizer
from src.utils import validate_normalization, setup_results_dir

# Setup de Logger para o Teste
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('INTEGRITY_TEST')

def test_integrity():
    logger.info("=" * 80)
    logger.info("INICIANDO TESTE DE INTEGRIDADE DO PIPELINE")
    logger.info("=" * 80)

    # Garante diretórios
    setup_results_dir()

    # --- 1. CARREGAMENTO E PREPARAÇÃO (PASSO 1) ---
    logger.info("[PASSO 1] Carregando dados e estatísticas...")
    df_juros = carregar_taxa_juros(PATHS['selic_data'])
    if df_juros is None:
        logger.error("Erro Crítico: Falha ao carregar Selic.")
        return False

    full_dataset, data_stats = criar_dataset_hibrido(
        caminho_pasta_opcoes=PATHS['raw_data'],
        df_juros=df_juros,
        seq_length=DATA_CONFIG['sequence_length']
    )
    
    if full_dataset is None:
        logger.error("Erro Crítico: Dataset não encontrado.")
        return False

    logger.info(f"Dataset carregado com {len(full_dataset)} amostras.")

    # --- 2. TESTE DE ESCALA (PASSO 2) ---
    logger.info("[PASSO 2] Validando escalas de entrada (QA)...")
    temp_loader = DataLoader(full_dataset, batch_size=128, shuffle=True)
    
    try:
        # Chama a função de validação que implementamos
        report = validate_normalization(temp_loader, data_stats, raise_on_critical=False)
        logger.info(f"Relatório de Escala:\n{report}")
        
        # Inspeção manual extra para o log de teste
        batch = next(iter(temp_loader))
        x_seq, x_phy, y, _, _, _ = batch
        
        logger.info(f"Estatísticas de Amostra (X_seq): mean={x_seq.mean():.4f}, std={x_seq.std():.4f}")
        logger.info(f"Estatísticas de Amostra (X_phy): mean={x_phy.mean():.4f}, std={x_phy.std():.4f}")
        logger.info(f"Estatísticas de Amostra (y): mean={y.mean():.4f}, std={y.std():.4f}")
        
    except Exception as e:
        logger.error(f"Erro ao validar escalas: {e}")
        return False

    # --- 3. TESTE DE SHAPES E MÓDULOS (PASSO 3) ---
    logger.info("[PASSO 3] Testando Forward Pass e Componentes de Loss...")
    
    # Prepara modelo para teste
    model_cfg = MODEL_CONFIG.copy()
    num_assets = len(data_stats.get('asset_map', {}))
    model_cfg['num_assets'] = max(num_assets, 10)
    
    model = DeepHestonHybrid(config=model_cfg, data_stats=data_stats)
    model.eval()
    
    batch = next(iter(temp_loader))
    x_seq, x_phy, y, x_time, weights, asset_ids = batch
    
    try:
        with torch.no_grad():
            outputs = model(x_seq, x_phy, asset_ids)
            
            # Valida shapes
            logger.info(f"Shape Saída Preço: {outputs['price'].shape} (Esperado: [B, 1])")
            # heston_params é uma tupla (nu, theta, kappa, xi, rho)
            logger.info(f"Parâmetros Heston detectados (count): {len(outputs['heston_params'])}")
            
            if outputs['price'].shape[0] != x_seq.shape[0]:
                logger.error("Erro: Batch size da saída não bate com a entrada.")
                return False
                
        # Teste de Loss Componentes no Trainer
        # Criamos um trainer minimalista para testar a função de compute_loss
        trainer = PINNTrainer(
            model=model,
            train_loader=temp_loader,
            val_loader=temp_loader,
            data_stats=data_stats,
            config=TRAINING_CONFIG
        )
        
        # O novo formato retorna 6 valores e exige weight_pde_curr
        loss_total, l_data, l_pde, l_bc, l_reg, _ = trainer.compute_loss(batch, weight_pde_curr=1.0)
        logger.info(f"Componentes da Loss: data={l_data.item():.4f}, pde={l_pde.item():.4f}, bc={l_bc.item():.4f}")
        
        if l_data is None or l_pde is None:
            logger.error("Erro: Componentes essenciais da loss (Data/PDE) ausentes.")
            return False

    except Exception as e:
        logger.error(f"Erro no teste de módulos: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    # --- 4. TREINAMENTO CURTO E PLOTS (PASSO 4) ---
    logger.info("[PASSO 4] Executando treinamento curto (2 épocas) para teste de plots...")
    
    # Sobrescreve config para ser ultra-rápido
    short_config = TRAINING_CONFIG.copy()
    short_config['epochs_per_phase'] = 2
    short_config['learning_rates'] = [5e-4]
    short_config['batch_size'] = 256
    
    trainer_short = PINNTrainer(
        model=model,
        train_loader=temp_loader,
        val_loader=temp_loader,
        data_stats=data_stats,
        config=short_config
    )
    
    try:
        trainer_short.train()
        logger.info("Treinamento curto finalizado.")
        
        # Tenta gerar os plots
        history_path = os.path.join(PATHS['results_dir'], 'training_history.csv')
        viz = Visualizer(
            model=model,
            history_path=history_path,
            val_loader=temp_loader,
            data_stats=data_stats,
            config=VIZ_CONFIG
        )
        
        logger.info("Gerando plots de diagnóstico...")
        viz.plot_all()
        
    except Exception as e:
        logger.error(f"Erro na fase de treinamento/plot: {e}")
        return False

    logger.info("=" * 80)
    logger.info("TESTE DE INTEGRIDADE FINALIZADO COM SUCESSO!")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    success = test_integrity()
    if not success:
        logger.error("O Teste de Integridade FALHOU.")
        sys.exit(1)
    else:
        sys.exit(0)
