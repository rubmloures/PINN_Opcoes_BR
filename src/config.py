# /src/config.py

import torch
import os

# --- Estrutura de Diretórios ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dados')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados')

PATHS = {
    'raw_data': os.path.join(DATA_DIR, 'brutos'),
    'processed_data': os.path.join(DATA_DIR, 'processados', 'dados_unificados.csv'),
    'selic_data': os.path.join(DATA_DIR, 'brutos', 'taxa_selic.csv'),
    'model_save_dir': os.path.join(RESULTS_DIR, 'modelo_final'),
    'plot_save_dir': os.path.join(RESULTS_DIR, 'plots'),
    'results_dir': RESULTS_DIR,            # Pasta raiz para resultados (onde o CSV será salvo)
}

# --- Configurações de Pré-processamento de Dados ---
DATA_CONFIG = {
    'min_moneyness': 0.8,
    'max_moneyness': 1.2,
    'num_samples': 1_000_000,  # Amostragem para evitar sobrecarga de memória
    'test_size': 0.2,
    'random_state': 42,
}

# --- Configurações da Arquitetura da Rede Neural ---
# 'INVERSE': Aprende a volatilidade implícita (σ) a partir dos preços de mercado.
# 'FORWARD': Precifica a opção com base na volatilidade (σ) fornecida.
PROBLEM_TYPE = 'INVERSE' 
# Para o projeto, a combinação mais poderosa e alinhada com os objetivos avançados é INVERSE + PAYOFF_INSPIRED

MODEL_CONFIG = {
    'architecture': 'PAYOFF_INSPIRED',  # Opções: 'ORIGINAL', 'PAYOFF_INSPIRED'
    'input_size': 5, # S, K, T, r, premium (para o problema inverso)
    'fourier_features': 128,
    'fourier_sigma': 5.0, # Controla a escala das features de Fourier
    'shallow_layers': [256, 64], # Camadas do componente "raso"
    'deep_layers': [256, 512, 512, 512, 512, 256], # Camadas do componente "profundo"
    'activation_fn': torch.nn.GELU(),
}

# Se o problema for 'FORWARD', a entrada não inclui o prêmio
if PROBLEM_TYPE == 'FORWARD':
    MODEL_CONFIG['input_size'] = 5 # S, K, T, r, vol

# --- Configurações de Treinamento ---
TRAINING_CONFIG = {
    # Configuração do device e tamanho do batch
    'device': 'cuda',                     # if torch.cuda.is_available() else 'cpu',
    'batch_size': 8192,
    'phy_batch_size': 8192,               # Batch size para os pontos de colocação da EDP

    # Configuração para Learning Rate e Stopping
    'use_adaptive_weights': True, 
    'learning_rates': [1e-4, 1e-5, 1e-6], # Lista de LRs para cada fase
    'epochs_per_phase': 5000,                 # TETO de segurança para épocas em uma fase
    'patience': 1000,                     # Paciência para o Early Stopping por fase
    'min_delta': 1e-7,

    # Amostragem por Importância
    'resample_every': 25,           # A cada 50 épocas, reavalia e seleciona novos pontos de colocação
    
    # Pesos do Curriculum Learning (ajustados dinamicamente por fase no trainer)
    'initial_pde_weight': 10.0,
    'initial_data_weight': 100.0,
    'final_pde_weight': 250.0,
    'final_data_weight': 10.0,
}

# --- Configurações de Visualização ---
VIZ_CONFIG = {
    'plot_loss': True,
    'plot_vol_surface': True,
    'plot_pde_residual': True,
    'plot_price_comparation': True,      
    'plot_moneyness_comparation': True,  
    'plot_greeks_comparation': True,     
    'plot_price_surfaces': True,         
    'plot_vol_smile': True,
    'plot_weights_history': True,              
}