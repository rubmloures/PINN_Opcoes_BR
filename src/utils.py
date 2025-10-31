# /src/utils.py

import pandas as pd
import os
from src.config import PATHS

def salvar_historico_treinamento(history: dict, nome_arquivo: str = "training_history.csv"):
    """
    Salva o dicionário de histórico de treinamento em um arquivo CSV.

    Args:
        history (dict): Dicionário contendo as listas de perdas.
        nome_arquivo (str): Nome do arquivo CSV de saída.
    """
    try:
        df_history = pd.DataFrame(history)
        
        # Garante que o diretório de resultados exista
        os.makedirs(PATHS['results_dir'], exist_ok=True)
        
        caminho_completo = os.path.join(PATHS['results_dir'], nome_arquivo)
        
        df_history.to_csv(caminho_completo, index_label='epoch')
        print(f"Histórico de treinamento salvo com sucesso em: {caminho_completo}")
    except Exception as e:
        print(f"Erro ao salvar o histórico de treinamento: {e}")

# Adicione aqui sua função send_telegram_message se desejar usá-la
# Exemplo:
# import requests
# def send_telegram_message(message):
#     TOKEN = "SEU_TOKEN_AQUI"
#     CHAT_ID = "SEU_CHAT_ID_AQUI"
#     url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
#     params = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
#     try:
#         response = requests.post(url, data=params)
#     except Exception as e:
#         print(f"Erro ao enviar mensagem para o Telegram: {e}")


