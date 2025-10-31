# /src/data_loader.py

import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def carregar_taxa_juros(caminho_selic: str) -> pd.DataFrame:
    """
    Carrega e pré-processa os dados da taxa Selic.

    Args:
        caminho_selic (str): Caminho para o arquivo CSV da taxa Selic.

    Returns:
        pd.DataFrame: DataFrame com as datas e a taxa de juros anualizada.
    """
    print(f"Carregando dados da taxa de juros de: {caminho_selic}")
    try:
        df_selic = pd.read_csv(
            caminho_selic,
            sep=',',
            decimal=',', # Reconhece a vírgula como separador decimal
            dtype={'valor': str} # Lê como string para tratar possíveis erros
        )
        df_selic['data'] = pd.to_datetime(df_selic['data'], dayfirst=True)
            # Limpa e converte os valores para float
        df_selic['valor'] = df_selic['valor'].str.replace(',', '.', regex=False).str.replace(' ', '').str.strip()
        df_selic['valor'] = pd.to_numeric(df_selic['valor'], errors='coerce')
        df_selic['taxa_anual'] = df_selic['valor'] / 100.0 # Converte para decimal
        df_selic = df_selic.set_index('data')[['taxa_anual']].sort_index()
        print("Dados da taxa de juros carregados com sucesso.")
        return df_selic
    except FileNotFoundError:
        print(f"ERRO: Arquivo da taxa Selic não encontrado em '{caminho_selic}'. Verifique o caminho.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao processar o arquivo da Selic: {e}")
        return None

def carregar_e_processar_dados_opcoes(caminho_dados_brutos: str, df_juros: pd.DataFrame) -> pd.DataFrame:
    """
    Carrega todos os arquivos CSV de opções, limpa, enriquece com a taxa de juros
    e unifica em um único DataFrame.

    Args:
        caminho_dados_brutos (str): Caminho para a pasta com os arquivos CSV dos ativos.
        df_juros (pd.DataFrame): DataFrame pré-processado com as taxas de juros.

    Returns:
        pd.DataFrame: DataFrame unificado e processado.
    """
    arquivos_opcoes = glob(os.path.join(caminho_dados_brutos, '*.csv'))
    arquivos_opcoes = [f for f in arquivos_opcoes if 'taxa_selic' not in f] # Exclui o arquivo da Selic

    if not arquivos_opcoes:
        print(f"Nenhum arquivo de opções encontrado em '{caminho_dados_brutos}'.")
        return pd.DataFrame()

    lista_df_ativos = []
    print(f"Encontrados {len(arquivos_opcoes)} arquivos de ativos para processar...")

    for arquivo in tqdm(arquivos_opcoes, desc="Processando arquivos de ativos"):
        try:
            nome_ativo = os.path.basename(arquivo).split('_')[0]
            df_ativo = pd.read_csv(arquivo, sep=',')
            
            # --- Limpeza e Formatação ---
            df_ativo['ativo'] = nome_ativo
            df_ativo['time'] = pd.to_datetime(df_ativo['time']).dt.date
            df_ativo['time'] = pd.to_datetime(df_ativo['time'])

            # --- Filtros Essenciais para Qualidade dos Dados ---
            # Remove dados inconsistentes
            df_ativo = df_ativo[df_ativo['premium'] > 0]
            df_ativo = df_ativo[df_ativo['days_to_maturity'] > 0]
            df_ativo = df_ativo[df_ativo['volatility'] > 0]
            df_ativo = df_ativo[df_ativo['strike'] > 0]
            df_ativo = df_ativo[df_ativo['spot_price'] > 0]

            # Foco em opções de compra (CALL) conforme o objetivo
            df_ativo = df_ativo[df_ativo['option_type'] == 'CALL']
            
            # Remove linhas com valores NaN que podem ter restado
            df_ativo.dropna(subset=['spot_price', 'strike', 'premium', 'days_to_maturity', 'volatility'], inplace=True)

            if df_ativo.empty:
                print(f"Aviso: Nenhum dado válido restante para o ativo {nome_ativo} após a filtragem.")
                continue

            # --- Merge com a Taxa de Juros ---
            # Usa merge_asof para encontrar a taxa de juros mais recente para cada data de opção
            df_ativo = pd.merge_asof(
                df_ativo.sort_values('time'),
                df_juros,
                left_on='time',
                right_index=True,
                direction='backward'
            )
            df_ativo.rename(columns={'taxa_anual': 'r'}, inplace=True)
            df_ativo.dropna(subset=['r'], inplace=True) # Garante que todas as opções tenham uma taxa

            lista_df_ativos.append(df_ativo)

        except Exception as e:
            print(f"Erro ao processar o arquivo {arquivo}: {e}")

    if not lista_df_ativos:
        print("Nenhum dado de opção foi carregado com sucesso.")
        return pd.DataFrame()
        
    print("Unificando todos os dataframes de ativos...")
    df_unificado = pd.concat(lista_df_ativos, ignore_index=True)

    # --- Engenharia de Features ---
    print("Realizando engenharia de features...")
    df_unificado['time_to_maturity'] = df_unificado['days_to_maturity'] / 252.0
    df_unificado['vol'] = df_unificado['volatility'] / 100.0
    df_unificado['moneyness'] = df_unificado['spot_price'] / df_unificado['strike']
    
    # Filtro de moneyness para focar em opções mais líquidas e relevantes
    df_unificado = df_unificado[(df_unificado['moneyness'] >= 0.8) & (df_unificado['moneyness'] <= 1.2)]

    print(f"Processamento concluído. DataFrame final com {len(df_unificado)} amostras.")
    
    return df_unificado

def salvar_dados_processados(df: pd.DataFrame, caminho_saida: str):
    """
    Salva o DataFrame processado em um arquivo CSV.
    
    Args:
        df (pd.DataFrame): DataFrame a ser salvo.
        caminho_saida (str): Caminho completo do arquivo de saída.
    """
    if not df.empty:
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        df.to_csv(caminho_saida, index=False)
        print(f"Dados unificados e processados salvos em: {caminho_saida}")
    else:
        print("Nenhum dado para salvar.")

# Exemplo de como usar este módulo no main.py
if __name__ == '__main__':
    # Define os caminhos relativos à estrutura do projeto
    CAMINHO_DADOS_BRUTOS = '../dados/brutos'
    CAMINHO_SELIC = os.path.join(CAMINHO_DADOS_BRUTOS, 'taxa_selic.csv')
    CAMINHO_SAIDA_PROCESSADO = '../dados/processados/dados_unificados.csv'

    # 1. Carregar a taxa de juros
    df_juros = carregar_taxa_juros(CAMINHO_SELIC)

    if df_juros is not None:
        # 2. Carregar e processar os dados das opções
        df_final = carregar_e_processar_dados_opcoes(CAMINHO_DADOS_BRUTOS, df_juros)
        
        # 3. Salvar o resultado
        salvar_dados_processados(df_final, CAMINHO_SAIDA_PROCESSADO)
        
        print("\n--- Amostra do DataFrame Final ---")
        print(df_final.head())
        print("\n--- Informações do DataFrame ---")
        df_final.info()