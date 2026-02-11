# /src/config.py

import torch
import os

# --- Estrutura de Diretórios ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dados')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados')

PATHS = {
    'raw_data': os.path.join(DATA_DIR, 'brutos'),
    'dividend_data': os.path.join(DATA_DIR, 'brutos', 'dividend_yields.csv'), 
    'selic_data': os.path.join(DATA_DIR, 'brutos', 'taxa_selic.csv'),
    'ibov_data': os.path.join(DATA_DIR, 'brutos', 'BOVA11.csv'),
    'model_save_dir': os.path.join(RESULTS_DIR, 'modelo_final'),
    'plot_save_dir': os.path.join(RESULTS_DIR, 'plots'),
    'results_dir': RESULTS_DIR,
}

# --- Configurações de Dados e Sequências ---
DATA_CONFIG = {
    'sequence_length': 30,      # Janela histórica (ex: 30 dias) para a LSTM
    'test_size': 0.2,
    'random_state': 42,
    'num_samples': None,
    # --- Ponderação ---
    'use_sample_weights': True, # Ativa ponderação por moneyness
    'min_moneyness': 0.7,       # Foco na região relevante
    'max_moneyness': 1.3,
}

# --- Configurações da Arquitetura Híbrida (DeepHeston) ---
MODEL_CONFIG = {
    # Configs da LSTM (O Analista)
    'lstm_input_size': 6,       # [log_ret, rolling_vol_20, ewma_vol, vol_parkinson, log_vol_fin, log_ret_ibov]
    'lstm_hidden_size': 64,     # Tamanho do vetor de estado do mercado
    'lstm_layers': 2,
    'lstm_dropout': 0.2,
    
    # --- Asset Embeddings ---
    'use_asset_embeddings': True,
    'num_assets': 20,           # Tamanho do vocabulário (estimativa segura, ajustado dinamicamente no loader se quiser)
    'asset_embedding_dim': 4,   # Dimensão do vetor latente por ativo

    # Configs da PINN (O Físico)
    # Input: 5 (S,K,T,r,q) + 5 (Parâmetros Heston: nu, theta, kappa, xi, rho)   
    # q = dividend yield
    # Total input da rede densa = 10
    'pinn_hidden_layers': 4,        # Camadas ocultas da PINN
    'pinn_neurons': 64,             # Neurônios por camada oculta
    'output_dim': 2,                # Preço e Volatilidade
    'activation': 'tanh',           # SiLU (Swish) para melhor fluxo de gradiente
    'dropout': 0.00,                # Dropout 

    # Fourier Features
    'use_fourier_features': True,   # Mapeamento para capturar altas frequências
    'fourier_features': 128,        # Dimensão da projeção (gera 256 features: sin/cos)
    'fourier_sigma': 1.0,           # Escala das frequências amostradas
}

# --- Configurações de Treinamento ---
TRAINING_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # Performance GPU
    'batch_size': 4096,       # Alto para maximizar throughput da GPU
    'phy_batch_size': 4096,   # Pontos de colocação da EDP
    
    # Pesos Adaptativos
    'use_adaptive_weights': False,  # Ativado para balancear Data vs PDE dinamicamente
    # Pesos Fixos Manuais (usados quando use_adaptive_weights=False)
    'weight_data': 1.0,          # Peso da loss de dados (normalizada)
    'weight_pde': 0.1,          # Peso da loss de física 
    
    # Curriculum
    'warmup_epochs': 20,        # Épocas iniciais com peso físico ZERO (só aprende dados)
    'rampup_epochs': 50,        # Épocas para subir o peso físico de 0 até o target
    
    # --- Curriculum Learning (Fases) ---
    # Lista de LRs: começa alto para exploração, diminui para refinamento
    'learning_rates': [1e-4, 5e-5, 1e-5, 1e-6], # Fases de LR decrescente 
    'epochs_per_phase': 5,                    # Teto de épocas por fase 300
    'patience': 2,                            # Paciência para acionar early stopping 50
    'min_delta': 1e-7,                          # Exige melhora real para continuar
    
    # --- Fine-Tuning (Especialização por Ativo) ---
    'finetune_epochs': 2,              # Épocas por ativo
    'finetune_learning_rate': 1e-4,     # LR específico para fine-tune
    'finetune_batch_size': 256,         # Batch menor para dados específicos do ativo
    'finetune_patience': 10,            # Early stopping mais permissivo

    # --- Pesos para Física Avançada (Literature-Based) ---
    'lambda_bc': 1.0,         # Peso para boundary conditions (Heston 1993, Beck et al. 2019)
    'lambda_reg': 0.01,       # Peso para regularização física dos parâmetros (Wang et al. 2020)
    
    # --- Amostragem por Importância ---
    'resample_every': 50,      # A cada n épocas, gera novos pontos (S, t) para a PDE
}

# --- Configurações de Visualização (Plotly - Interativo) ---
VIZ_CONFIG = {
    # Validação Física - Histórico
    'plot_loss_convergence': True,
    'plot_weights_history': True,            
    # Validação de Precificação
    'plot_price_scatter': True,
    'plot_residuals': True,
    'plot_error_by_moneyness': True,            
    # Superfícies e Gregas
    'plot_price_surface': True,
    'plot_delta_surface': True,
    'plot_pde_residual': True,            
    # Validação LSTM
    'plot_heston_params': True,            
    # Placeholders compatibilidade)
    'plot_premium_time': True,
    'plot_model_vs_bs': True,
    'plot_dist_overlay': True,
    'plot_error_distribution': True,
    'plot_error_heatmap': True,
    'plot_vol_smile': True,
    'plot_latent_vol_evolution': True,
    'plot_premium_by_moneyness_time': True,
}


#VIZ_CONFIG = {
    #'run_benchmark_first': True,             # Executa benchmark_calibration antes dos plots
    #'export_plots_png': True,                # Exporta plots também em PNG (requer kaleido)
#    
    ## ===== SEÇÃO A: VALIDAÇÃO DO LSTM vs GROUND TRUTH =====
    #'plot_lstm_heston_params': True,         # Scatter LSTM vs Ground Truth (parâmetros Heston)
    #'plot_lstm_residuals': True,             # Histograma de resíduos (erros dos parâmetros)
    #'plot_lstm_timeseries': True,            # Série temporal: parâmetros reais vs previstos
#    
    ## ===== SEÇÃO B: VALIDAÇÃO DA PINN =====
    #'plot_loss_convergence': True,           # Convergência: Loss Data, PDE, BC, IC
    #'plot_overfitting_detection': True,      # Learning Curves (Train vs Validation)
#    
    ## ===== SEÇÃO C: VALIDAÇÃO DO MODELO INTEGRADO =====
    #'plot_pricing_error': True,              # MAPE, MAE, RMSE, R² de precificação
    #'plot_prediction_intervals': True,       # Intervalos de confiança (bandas de predição)
    #'plot_greeks_surface': True,             # Superfícies 3D de Gregas (Delta, Gamma, Vega)
#    
    ## ===== SEÇÃO D: FINE-TUNING PROGRESS =====
    #'plot_finetuning': True,                 # Progresso do fine-tuning por ativo
#}