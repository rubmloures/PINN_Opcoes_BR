# /main.py

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import json
# Importa as configurações e os módulos que criamos
from src.config import PATHS, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG
from src.data_loader import carregar_taxa_juros, carregar_e_processar_dados_opcoes
from src.model import PINN_BlackScholes
from src.trainer import PINNTrainer
from src.visualization import Visualizer

def run_pipeline():
    """
    Função principal que executa todo o pipeline do projeto.
    """
    # --- 1. Carregamento e Preparação dos Dados ---
    print("Iniciando pipeline: Fase 1 - Carregamento de Dados")
    df_juros = carregar_taxa_juros(PATHS['selic_data'])
    if df_juros is None:
        return # Encerra se não conseguir carregar os juros

    df_opcoes = carregar_e_processar_dados_opcoes(PATHS['raw_data'], df_juros)
    if df_opcoes.empty:
        print("Pipeline encerrado: Nenhum dado de opção para processar.")
        return

    # --- 2. Normalização e Preparação dos Tensores ---
    print("\nFase 2 - Preparando dados para o PyTorch")
    
    # Amostragem para limitar o uso de memória, se configurado
    if len(df_opcoes) > DATA_CONFIG['num_samples']:
        print(f"Realizando amostragem de {DATA_CONFIG['num_samples']} amostras...")
        df_opcoes = df_opcoes.sample(n=DATA_CONFIG['num_samples'], random_state=DATA_CONFIG['random_state'])
        
    # Guardar estatísticas para desnormalização dentro do modelo
    data_stats = {
        'S_min': df_opcoes['spot_price'].min(), 'S_max': df_opcoes['spot_price'].max(),
        'K_min': df_opcoes['strike'].min(), 'K_max': df_opcoes['strike'].max(),
        'T_max': df_opcoes['time_to_maturity'].max(),
    }
    stats_path = os.path.join(PATHS['model_save_dir'], 'data_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(data_stats, f, indent=4)
    print(f"Estatísticas de normalização salvas em: {stats_path}")
    
    # Normalização das features de entrada
    df_opcoes['S_norm'] = (df_opcoes['spot_price'] - data_stats['S_min']) / (data_stats['S_max'] - data_stats['S_min'])
    df_opcoes['K_norm'] = (df_opcoes['strike'] - data_stats['K_min']) / (data_stats['K_max'] - data_stats['K_min'])
    df_opcoes['T_norm'] = df_opcoes['time_to_maturity'] / data_stats['T_max']
    
    # Amostragem estratificada baseada em 'moneyness'
    print("Criando categorias de 'moneyness' para amostragem estratificada...")
    # 1. Define os limites para cada categoria
    # OTM: < 0.95, ATM: 0.95 a 1.05, ITM: > 1.05
    bins = [0, 0.95, 1.05, np.inf]
    labels = ['OTM', 'ATM', 'ITM']
    # 2. Cria a nova coluna categórica no DataFrame
    df_opcoes['moneyness_category'] = pd.cut(df_opcoes['moneyness'], bins=bins, labels=labels)

    print("Distribuição das categorias:")
    print(df_opcoes['moneyness_category'].value_counts(normalize=True))

    # Define as colunas de entrada para o tensor
    # Para o problema INVERSO, o prêmio é uma das entradas
    input_features = ['S_norm', 'K_norm', 'T_norm', 'r', 'premium']
    
    X = df_opcoes[input_features].values
    y = df_opcoes[['premium']].values # O alvo da perda de dados ainda é o prêmio
    
    # Armazena a coluna de estratificação
    stratify_col = df_opcoes['moneyness_category']

    # Divisão em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=DATA_CONFIG['test_size'], 
        random_state=DATA_CONFIG['random_state'],
        stratify=stratify_col  
    )

    # Criação dos DataLoaders do PyTorch
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False, num_workers=0)

    print(f"Dados prontos: {len(train_dataset)} amostras de treino, {len(val_dataset)} amostras de validação.")

    # --- 3. Instanciação e Treinamento do Modelo ---
    print("\nFase 3 - Configurando e iniciando o treinamento do modelo")
    
    pinn_model = PINN_BlackScholes(config=MODEL_CONFIG, data_stats=data_stats)
    
    trainer = PINNTrainer(
        model=pinn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        data_stats=data_stats,
        config=TRAINING_CONFIG
    )
    
    trainer.train()

    # --- 4. Visualização dos Resultados (a ser implementado) ---
    print("\nFase 4 - Gerando visualizações dos resultados")
    # 1. Caminho para o arquivo de histórico.
    history_file_path = os.path.join(PATHS['results_dir'], 'training_history.csv')
    # 2. Instancia do Visualizer 
    viz = Visualizer(
        model=pinn_model, 
        history_path=history_file_path, 
        val_loader=val_loader,
        data_stats=data_stats, 
        config=VIZ_CONFIG
    ) 
    viz.plot_all()
    print("Plots salvos em:", PATHS['plot_save_dir'])

if __name__ == '__main__':
    run_pipeline()