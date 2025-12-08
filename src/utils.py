# /src/utils.py

import pandas as pd
import os
from src.config import PATHS
from src.logger import get_logger

# Configurar logger
logger = get_logger('PINN_Utils')

def salvar_historico_treinamento(history: dict, results_dir: str = None):
    """
    Salva o histórico de treinamento em CSV de forma segura,
    lidando com listas de tamanhos diferentes.
    
    Args:
        history: Dicionário com histórico de treinamento
        results_dir: Diretório para salvar (opcional, usa PATHS['results_dir'] se não informado)
    """
    try:
        if results_dir is None:
            results_dir = PATHS['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'training_history.csv')
        
        # Encontra o comprimento máximo entre as listas
        max_len = max(len(v) for v in history.values() if isinstance(v, list))
        
        # Normaliza as listas (preenche com NaN se faltar dado no final)
        history_normalized = {}
        for k, v in history.items():
            if isinstance(v, list):
                if len(v) < max_len:
                    # Estende com None/NaN
                    v = v + [None] * (max_len - len(v))
                history_normalized[k] = v
        
        df = pd.DataFrame(history_normalized)
        df.to_csv(save_path, index=False)
        logger.info(f"Histórico salvo com sucesso em: {save_path}")
        
    except Exception as e:
        logger.error(f"Erro crítico ao salvar histórico: {e}")
        # Tenta salvar backup cru
        try:
            pd.DataFrame.from_dict(history, orient='index').transpose().to_csv(save_path + ".bak")
            logger.info("Backup salvo.")
        except:
            pass

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


