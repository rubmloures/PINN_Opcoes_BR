# /src/data_loader.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from src.config import DATA_CONFIG,PATHS
from src.logger import get_logger

# Configurar logger
logger = get_logger('PINN_DataLoader')

def carregar_taxa_juros(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega a taxa Selic de um CSV e retorna um DataFrame indexado por data.
    """
    if not os.path.exists(caminho_arquivo):
        logger.warning(f"Arquivo de juros não encontrado: {caminho_arquivo}")
        return None
    try:
        df_selic = pd.read_csv(caminho_arquivo)
        # Lógica de tratamento de colunas (mantida do original)
        col_data = 'data' if 'data' in df_selic.columns else 'time'
        col_valor = 'valor' if 'valor' in df_selic.columns else df_selic.columns[1]
        
        df_selic['data_only'] = pd.to_datetime(df_selic[col_data], dayfirst=True, errors='coerce').dt.normalize()
        
        if df_selic[col_valor].dtype == object:
            df_selic[col_valor] = df_selic[col_valor].astype(str).str.replace('"', '').str.replace(',', '.')
        
        df_selic['r'] = pd.to_numeric(df_selic[col_valor], errors='coerce')
        if df_selic['r'].mean() > 1.0:
            df_selic['r'] = df_selic['r'] / 100.0
            
        return df_selic[['data_only', 'r']].dropna().drop_duplicates('data_only').set_index('data_only')
    except Exception as e:
        logger.error(f"Erro ao ler Selic: {e}")
        return None

def carregar_dividendos(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega Dividend Yields e remove timezone para compatibilidade.
    """
    if not os.path.exists(caminho_arquivo):
        logger.warning(f"Arquivo de dividendos não encontrado: {caminho_arquivo}")
        return None
    try:
        df = pd.read_csv(caminho_arquivo)
        
        # --- CORREÇÃO DE TIMEZONE ---
        # 1. Converte para datetime (utc=True garante parsing unificado)
        # 2. .dt.tz_localize(None) remove a info de fuso horário, tornando-o 'naive' igual ao df_full
        # 3. .dt.normalize() zera as horas para bater apenas a data
        df['data_only'] = pd.to_datetime(df['data_only'], utc=True).dt.tz_localize(None).dt.normalize()
        
        df = df.sort_values('data_only')
        
        return df[['data_only', 'ativo', 'Dividend_Yield']]
    except Exception as e:
        logger.error(f"Erro ao ler Dividendos: {e}")
        return None
    
def calcular_peso_amostra(moneyness: float, price: float) -> float:
    """
    Calcula peso para o treinamento baseado na importância financeira.
    - Maior peso para ATM (moneyness ~ 1.0)
    - Penaliza OTMs muito baratas que distorcem o gradiente.
    """
    if not DATA_CONFIG.get('use_sample_weights', False):
        return 1.0
        
    # Distância do dinheiro (quanto mais perto de 1, mais importante)
    dist_atm = abs(moneyness - 1.0)
    
    # Fórmula de decaimento suave: Opções ATM têm peso ~10x maior que deep OTM
    weight = 1.0 / (dist_atm + 0.1)
    
    # (Opcional) Reduzir peso se o preço for muito baixo (ruído de microestrutura)
    if price < 0.10:
        weight *= 0.5
        
    return weight

def preparar_dados_lstm(caminho_pasta: str, df_selic: pd.DataFrame, sequence_length: int = 30):
    """
    Gera sequências para LSTM e inputs para PINN, incluindo pesos de amostragem.
    """
    arquivos = [os.path.join(caminho_pasta, f) for f in os.listdir(caminho_pasta) if f.endswith('.csv')]
    logger.info(f"Lendo arquivos de opções em: {caminho_pasta}")
    
    sequences = []
    pinn_inputs = []
    targets = []
    timestamps = []
    sample_weights = [] # Novo vetor de pesos
    
    # Cache de ativos encontrados
    ativos_encontrados = set()

    for arq in tqdm(arquivos, desc="Gerando sequências temporais"):
        df = pd.read_csv(arq)
        # Filtros
        if 'option_type' in df.columns: 
            df = df[df['option_type'] == 'CALL']
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        # Filtro de Moneyness (Importante para estabilidade)
        if 'moneyness' in df.columns:
             df = df[(df['moneyness'] >= DATA_CONFIG.get('min_moneyness', 0.8)) & 
                     (df['moneyness'] <= DATA_CONFIG.get('max_moneyness', 1.2))]

        symbol = df['symbol'].iloc[0] if not df.empty else 'UNKNOWN'
        ativos_encontrados.add(symbol[:4])

        # Merge com Selic
        df['data_only'] = df['time'].dt.normalize()
        if df_selic is not None:
            df = df.merge(df_selic, left_on='data_only', right_index=True, how='left')
            df['r'] = df['r'].fillna(0.11) # Fallback Selic
        else:
            df['r'] = 0.11

        # Normalização de Features da LSTM
        features = ['spot_price', 'strike', 'days_to_maturity', 'r', 'premium']
        feature_data = df[features].values.astype(np.float32)
        
        # Normalização MinMax Simples (Local por ativo para preservar tendência)
        for i in range(feature_data.shape[1]):
            col = feature_data[:, i]
            if col.max() != col.min():
                feature_data[:, i] = (col - col.min()) / (col.max() - col.min())

        # Janelamento
        for idx in range(sequence_length, len(df)):
            row = df.iloc[idx]
            
            # Evita dias com vencimento muito curto (ruído numérico na PDE)
            if row['days_to_maturity'] > 5:
                sequences.append(feature_data[idx-sequence_length : idx])
                
                # Inputs físicos: [S, K, tau, r]
                # Tau anualizado
                pinn_inputs.append([row.spot_price, row.strike, row.days_to_maturity/252, row.r])
                
                targets.append(row.premium)
                timestamps.append(row.time.timestamp())
                
                # Calcula Peso
                w = calcular_peso_amostra(row.moneyness, row.premium) if 'moneyness' in df.columns else 1.0
                sample_weights.append(w)

    if not targets: 
        return None, None
        
    logger.info(f"Ativos encontrados: {ativos_encontrados}")

    # Conversão para Numpy
    X_seq = np.array(sequences, dtype=np.float32)
    X_phy = np.array(pinn_inputs, dtype=np.float32)
    y = np.array(targets, dtype=np.float32).reshape(-1, 1)
    
    # Normalização do Target (Preço / Strike)
    strike_vals = X_phy[:, 1:2] + 1e-8
    y_normalized = y / strike_vals

    X_time = np.array(timestamps, dtype=np.float64).reshape(-1, 1)
    weights = np.array(sample_weights, dtype=np.float32).reshape(-1, 1)
    
    # Normalização dos pesos (média 1.0) para não alterar a escala da Loss
    weights = weights / weights.mean()

    logger.info(f"Dataset Final: {len(y_normalized)} amostras.")

    # Extrair asset_ids (todos do mesmo ativo nesta função)
    asset_map = {ativo: 0}  # Apenas um ativo nesta função
    X_asset = np.zeros(len(y_normalized), dtype=np.int64)

    # Estatísticas para normalização inversa no modelo
    S_raw, K_raw, T_raw = X_phy[:, 0], X_phy[:, 1], X_phy[:, 2]
    data_stats = {
        'S_min': float(S_raw.min()), 'S_max': float(S_raw.max()),
        'K_min': float(K_raw.min()), 'K_max': float(K_raw.max()),
        'T_max': float(T_raw.max()),
        'asset_map': asset_map  # Novo: mapa de ativos
    }

    # Criação do TensorDataset com 6 componentes (uniforme com criar_dataset_hibrido)
    dataset = TensorDataset(
        torch.from_numpy(X_seq),
        torch.from_numpy(X_phy),
        torch.from_numpy(y_normalized),
        torch.from_numpy(X_time),
        torch.from_numpy(weights),
        torch.from_numpy(X_asset)  # Elemento 5: Asset ID
    )

    return dataset, data_stats

def carregar_ibov(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega BOVA11 (índice Bovespa) e retorna DataFrame com log retorno do mercado.
    
    Args:
        caminho_arquivo: Caminho para BOVA11.csv
        
    Returns:
        DataFrame indexado por data com coluna 'log_ret_ibov'
    """
    if not os.path.exists(caminho_arquivo):
        logger.warning(f"Arquivo IBOV não encontrado: {caminho_arquivo}")
        return None
    
    try:
        # IMPORTANTE: skiprows=1 pula a linha 'sep=;' que aparece em alguns arquivos
        df_ibov = pd.read_csv(caminho_arquivo, sep=';', decimal=',', low_memory=False, skiprows=1)
        
        # Procura coluna de preço (tenta múltiplas opções)
        close_col = None
        for col_candidate in ['acao_close_ajustado', 'close', 'Close', 'CLOSE', 'adj_close', 'Adj Close']:
            if col_candidate in df_ibov.columns:
                close_col = col_candidate
                break
        
        if close_col is None:
            logger.warning(f"Coluna de preço não encontrada em BOVA11.csv. Colunas disponíveis: {df_ibov.columns.tolist()}")
            return None
            
        # Procura coluna de data (tenta múltiplas opções)
        date_col = None
        for col_candidate in ['time', 'data', 'Data', 'DATE', 'date', 'Date']:
            if col_candidate in df_ibov.columns:
                date_col = col_candidate
                break
        
        if date_col is None:
            logger.warning(f"Coluna de data não encontrada em BOVA11.csv. Colunas disponíveis: {df_ibov.columns.tolist()}")
            return None
        
        # Converte data para datetime normalizado
        df_ibov['data_only'] = pd.to_datetime(df_ibov[date_col], dayfirst=True, errors='coerce').dt.normalize()
        
        # Calcula retorno logarítmico
        df_ibov = df_ibov.sort_values('data_only')
        df_ibov[close_col] = pd.to_numeric(df_ibov[close_col], errors='coerce')
        df_ibov['log_ret_ibov'] = np.log(df_ibov[close_col] / df_ibov[close_col].shift(1))
        df_ibov['log_ret_ibov'] = df_ibov['log_ret_ibov'].fillna(0.0)
        
        logger.info(f"IBOV carregado com sucesso: {len(df_ibov)} registros, coluna de preço: {close_col}")
        return df_ibov[['data_only', 'log_ret_ibov']].drop_duplicates('data_only').set_index('data_only')
    
    except Exception as e:
        logger.error(f"Erro ao ler BOVA11: {e}")
        return None


def _calcular_features_ativo(df_ativo: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula 6 features para entrada da LSTM:
    1. log_ret - Retorno logarítmico do ativo
    2. rolling_vol_20 - Volatilidade histórica (20 dias)
    3. ewma_vol - Volatilidade exponencial
    4. vol_parkinson - Volatilidade High-Low
    5. log_vol_fin - Log do volume financeiro
    (6ª feature log_ret_ibov é adicionada no merge em criar_dataset_hibrido)
    """
    df = df_ativo.copy().sort_values('time')
    
    # Feature 1: Retorno logarítmico
    df['log_ret'] = np.log(df['spot_price'] / df['spot_price'].shift(1))
    df['log_ret'] = df['log_ret'].fillna(0.0)
    
    # Feature 2: Volatilidade histórica (rolling 20 dias)
    df['rolling_vol_20'] = df['log_ret'].rolling(window=20).std()
    df['rolling_vol_20'] = df['rolling_vol_20'].fillna(0.0)
    
    # Feature 3: Volatilidade exponencial (EWMA com span=20)
    # Nota: Se 'ewma_vol' não existir nos dados brutos, calculamos a partir de log_ret
    if 'ewma_vol' not in df.columns:
        df['ewma_vol'] = df['log_ret'].ewm(span=20, adjust=False).std()
        df['ewma_vol'] = df['ewma_vol'].fillna(0.0)
    
    # Feature 4: Volatilidade Parkinson (High-Low) se disponível
    if 'high_price' in df.columns and 'low_price' in df.columns:
        df['vol_parkinson'] = np.sqrt(np.log(df['high_price'] / df['low_price'])**2 / (4 * np.log(2)))
        df['vol_parkinson'] = df['vol_parkinson'].fillna(0.0)
    else:
        df['vol_parkinson'] = 0.0
    
    # Feature 5: Log do volume financeiro
    if 'acao_vol_fin' in df.columns:
        df['log_vol_fin'] = np.log1p(df['acao_vol_fin'])
    elif 'volume' in df.columns:
        df['log_vol_fin'] = np.log1p(df['volume'])
    else:
        df['log_vol_fin'] = 0.0
    
    df['log_vol_fin'] = df['log_vol_fin'].fillna(0.0)
    
    # Normalização MinMax local para ewma_vol, vol_parkinson, log_vol_fin
    for col in ['ewma_vol', 'vol_parkinson', 'log_vol_fin']:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0
    
    return df

def criar_dataset_hibrido(caminho_pasta_opcoes: str, df_juros: pd.DataFrame, seq_length: int = 30):
    """
    Gera dataset com: (X_seq, X_phy, y, X_time, weights).
    X_time contém o timestamp Unix para reconstrução temporal nos plots.
    X_seq agora possui 6 features: [log_ret, rolling_vol_20, ewma_vol, vol_parkinson, log_vol_fin, log_ret_ibov]
    """
    lista_dfs = []
    if not os.path.exists(caminho_pasta_opcoes):
        return None, None

    logger.info(f"Lendo arquivos de opções em: {caminho_pasta_opcoes}")
    nome_arquivo_selic = "taxa_selic.csv" 

    # === Carregamento do IBOV (mercado) ===
    df_ibov = carregar_ibov(PATHS.get('ibov_data', os.path.join(os.path.dirname(caminho_pasta_opcoes), 'BOVA11.csv')))
    if df_ibov is not None:
        logger.info(f"IBOV carregado com {len(df_ibov)} datas")
    else:
        logger.warning("Usando retorno de mercado = 0 (IBOV não disponível)")

    for arquivo in os.listdir(caminho_pasta_opcoes):
        if arquivo.endswith('.csv') and arquivo != nome_arquivo_selic:
            path_completo = os.path.join(caminho_pasta_opcoes, arquivo)
            try:
                # Leitura com separadores PT-BR
                # IMPORTANTE: skiprows=1 pula a linha 'sep=;' que aparece em alguns arquivos
                df_head = pd.read_csv(path_completo, nrows=2, sep=';', decimal=',', skiprows=1)
                if 'time' not in df_head.columns: continue
                
                df_temp = pd.read_csv(path_completo, sep=';', decimal=',', skiprows=1)
                df_temp['time'] = pd.to_datetime(df_temp['time'], errors='coerce', utc=True).dt.tz_localize(None)
                
                if 'ativo' not in df_temp.columns:
                    df_temp['ativo'] = df_temp['symbol'].str[:4] if 'symbol' in df_temp.columns else 'UNKNOWN'
                
                cols_req = ['time', 'spot_price', 'strike', 'premium', 'days_to_maturity', 'ativo']
                if all(c in df_temp.columns for c in cols_req):
                    lista_dfs.append(df_temp[cols_req])
            except: 
                continue
            
    if not lista_dfs: 
        return None, None

    df_full = pd.concat(lista_dfs, ignore_index=True)
    # Filtro Temporal Rigoroso (Impede vazamento de dados futuros ou lixo)
    df_full = df_full[df_full['time'].dt.year <= 2023]
    # Filtro de Consistência (Remove prêmios zerados/negativos)
    if 'premium' in df_full.columns:
        df_full = df_full[df_full['premium'] > 0.0]
    if not df_full.empty:
        data_ini = df_full['time'].min()
        data_fim = df_full['time'].max()
        logger.info(f"Intervalos de dados carregados, com {len(df_full)} registos.") 
        logger.info(f"Data de início {data_ini} ")
        logger.info(f"Data do fim {data_fim}")

    else:
        logger.error("Dataset vazio após filtros de data/consistência!")
        return None, None

    logger.info(f"Ativos encontrados: {df_full['ativo'].unique()}")

    # 1. Merge Juros
    df_full['data_only'] = df_full['time'].dt.normalize()
    if df_juros is not None:
        df_full = pd.merge(df_full, df_juros, on='data_only', how='left')
        df_full['r'] = df_full['r'].ffill().bfill().fillna(0.10)
    else:
        df_full['r'] = 0.10

    # 2. Merge Dividendos 
    df_divs = carregar_dividendos(PATHS['dividend_data'])
    if df_divs is not None:
        # Merge por data E ativo
        df_full = pd.merge(df_full, df_divs, on=['data_only', 'ativo'], how='left')
        
        # Tratamento do GAP: Preencher para frente e para trás POR ATIVO
        # Se um ativo não tiver dados, assume 0.0
        df_full['Dividend_Yield'] = df_full.groupby('ativo')['Dividend_Yield'].ffill().bfill().fillna(0.0)
    else:
        logger.warning("Dados de dividendos não disponíveis. Assumindo q=0.")
        df_full['Dividend_Yield'] = 0.0

    # 3. Merge IBOV (mercado)
    if df_ibov is not None:
        df_full = pd.merge(df_full, df_ibov, on='data_only', how='left')
        df_full['log_ret_ibov'] = df_full['log_ret_ibov'].ffill().bfill().fillna(0.0)
    else:
        df_full['log_ret_ibov'] = 0.0

    # 4. Mapeamento de Ativos para Embedding 
    ativos_unicos = sorted(df_full['ativo'].unique())
    asset_map = {ativo: i for i, ativo in enumerate(ativos_unicos)}
    logger.info(f"Asset Map: {asset_map}")

    logger.info("Gerando sequências...")
    sequences, pinn_inputs, targets, timestamps = [], [], [], []
    asset_ids_list = [] 
    sample_weights_list = []
    
    for nome_ativo, df_grupo in df_full.groupby('ativo'):
        df_grupo = df_grupo.sort_values('time')
        asset_id = asset_map[nome_ativo]
        
        df_asset = df_grupo[['time', 'spot_price']].drop_duplicates('time').copy()
        if len(df_asset) < seq_length + 20: continue
            
        df_asset = _calcular_features_ativo(df_asset)
        
        # Merge com IBOV para completar as 6 features
        if df_ibov is not None:
            df_asset['data_only'] = df_asset['time'].dt.normalize()
            df_asset = pd.merge(df_asset, df_ibov, on='data_only', how='left')
            df_asset['log_ret_ibov'] = df_asset['log_ret_ibov'].fillna(0.0)
        else:
            df_asset['log_ret_ibov'] = 0.0
        
        asset_dates = pd.to_datetime(df_asset['time']).tolist()
        
        # === Seleção das 6 colunas de features para LSTM ===
        cols_lstm = ['log_ret', 'rolling_vol_20', 'ewma_vol', 'vol_parkinson', 'log_vol_fin', 'log_ret_ibov']
        asset_feats = df_asset[cols_lstm].values.astype(np.float32)
        
        # Verificação de NaN/Inf antes de criar sequências
        if np.any(np.isnan(asset_feats)) or np.any(np.isinf(asset_feats)):
            logger.warning(f"Ativo {nome_ativo} contém NaN/Inf. Preenchendo com 0...")
            asset_feats = np.nan_to_num(asset_feats, nan=0.0, posinf=0.0, neginf=0.0)
        
        date_map = {ts: i for i, ts in enumerate(asset_dates)}
        
        for row in df_grupo.itertuples():
            idx = date_map.get(row.time)
            if idx is not None and idx >= seq_length:
                sequences.append(asset_feats[idx-seq_length : idx])
                
                # Input Físico: [S, K, tau, r, q]
                pinn_inputs.append([
                    row.spot_price, 
                    row.strike, 
                    row.days_to_maturity/252.0, 
                    row.r,
                    row.Dividend_Yield # q
                ])
                
                targets.append(row.premium)
                timestamps.append(row.time.timestamp())
                asset_ids_list.append(asset_id)
                
                w = calcular_peso_amostra(row.spot_price/row.strike, row.premium)
                sample_weights_list.append(w)

    if not targets: return None, None

    X_seq = np.array(sequences, dtype=np.float32)
    X_phy = np.array(pinn_inputs, dtype=np.float32)
    y = np.array(targets, dtype=np.float32).reshape(-1, 1)
    X_asset = np.array(asset_ids_list, dtype=np.int64)
    
    # === STEP 1: Salvar estatísticas RAW ANTES da normalização ===
    S_raw, K_raw, T_raw, r_raw, q_raw = X_phy[:, 0], X_phy[:, 1], X_phy[:, 2], X_phy[:, 3], X_phy[:, 4]
    
    # Calcula moneyness original para verificação
    moneyness_raw = S_raw / (K_raw + 1e-8)
    
    data_stats = {
        # Z-Score parameters para INPUTS
        'S_mean': float(S_raw.mean()), 'S_std': float(S_raw.std()),
        'K_mean': float(K_raw.mean()), 'K_std': float(K_raw.std()),
        'T_mean': float(T_raw.mean()), 'T_std': float(T_raw.std()),
        'r_mean': float(r_raw.mean()), 'r_std': float(r_raw.std()),
        'q_mean': float(q_raw.mean()), 'q_std': float(q_raw.std()),
        
        # Min/Max para boundary conditions e payoff físico
        'S_min': float(S_raw.min()), 'S_max': float(S_raw.max()),
        'K_min': float(K_raw.min()), 'K_max': float(K_raw.max()),
        'T_max': float(T_raw.max()),
        
        # Moneyness statistics (para verificações)
        'moneyness_mean': float(moneyness_raw.mean()),
        'moneyness_std': float(moneyness_raw.std()),
        
        # Asset mapping
        'asset_map': asset_map
    }
    
    # === STEP 2: Normalização TARGET (P/K) - Mantém Positividade ===
    y_raw_premium = y.copy()  # Backup para logs
    y = y / (X_phy[:, 1:2] + 1e-8)  # Normaliza: y_norm = P/K
    
    # Salva estatísticas do target normalizado
    data_stats['y_mean'] = float(y.mean())
    data_stats['y_std'] = float(y.std())
    data_stats['y_min'] = float(y.min())
    data_stats['y_max'] = float(y.max())
    
    # === STEP 3: Normalização Z-Score dos INPUTS ===
    # Preserva relações lineares (S - K) que representam valor intrínseco
    X_phy[:, 0] = (X_phy[:, 0] - data_stats['S_mean']) / (data_stats['S_std'] + 1e-8)  # S
    X_phy[:, 1] = (X_phy[:, 1] - data_stats['K_mean']) / (data_stats['K_std'] + 1e-8)  # K
    X_phy[:, 2] = (X_phy[:, 2] - data_stats['T_mean']) / (data_stats['T_std'] + 1e-8)  # T
    X_phy[:, 3] = (X_phy[:, 3] - data_stats['r_mean']) / (data_stats['r_std'] + 1e-8)  # r
    X_phy[:, 4] = (X_phy[:, 4] - data_stats['q_mean']) / (data_stats['q_std'] + 1e-8)  # q
    
    # === STEP 4: Verificação de Sanidade ===
    logger.info(f"Dataset Final: {len(y)} amostras.")
    logger.info(f"Target (P/K) - Mean: {data_stats['y_mean']:.4f}, Std: {data_stats['y_std']:.4f}")
    logger.info(f"Moneyness Raw - Mean: {data_stats['moneyness_mean']:.4f}, Std: {data_stats['moneyness_std']:.4f}")
    logger.info(f"S_norm - Mean: {X_phy[:, 0].mean():.4f}, Std: {X_phy[:, 0].std():.4f}")
    logger.info(f"K_norm - Mean: {X_phy[:, 1].mean():.4f}, Std: {X_phy[:, 1].std():.4f}")
    
    X_time = np.array(timestamps, dtype=np.float64).reshape(-1, 1)
    weights = np.array(sample_weights_list, dtype=np.float32).reshape(-1, 1)
    weights = weights / (weights.mean() + 1e-8)
    
    # === STEP 5: Validação e Sincronização de Asset Map ===
    expected_assets = set(ativos_unicos)
    saved_assets = set(data_stats.get('asset_map', {}).keys())
    
    if expected_assets != saved_assets:
        logger.warning(
            f"Asset mismatch detectado:\n"
            f"  Novo dataset: {expected_assets}\n"
            f"  Saved asset_map: {saved_assets}\n"
            f"  Atualizando asset_map..."
        )
        # Remapear com novos IDs
        data_stats['asset_map'] = asset_map
    
    logger.info(f"Asset map final: {data_stats['asset_map']}")
    
    # Retorna dataset com 6 elementos (adicionado Asset ID)
    return TensorDataset(
        torch.from_numpy(X_seq), 
        torch.from_numpy(X_phy), 
        torch.from_numpy(y),
        torch.from_numpy(X_time),
        torch.from_numpy(weights),
        torch.from_numpy(X_asset) # Elemento 5
    ), data_stats

