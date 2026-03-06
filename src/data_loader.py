# /src/data_loader.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from src.config import DATA_CONFIG, PATHS
from src.utils import salvar_dataset_integridade
from src.logger import get_logger

# Configurar logger
logger = get_logger('PINN_DataLoader')

def carregar_taxa_juros(caminho_arquivo: str) -> dict:
    """
    Carrega a taxa Selic de forma ultra-robusta (sem depender do auto-parsing do Pandas).
    Garante que o formato DD/MM/YYYY seja sempre convertido para YYYY-MM-DD.
    """
    if not os.path.exists(caminho_arquivo):
        logger.warning(f"Arquivo de juros não encontrado: {caminho_arquivo}")
        return {}
    
    lookup = {}
    total_linhas = 0
    try:
        import csv
        with open(caminho_arquivo, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                total_linhas += 1
                if len(row) < 2: continue
                
                # Limpeza e parsing da data
                dt_raw = row[0].strip().replace('"', '').replace("'", "")
                dt_parts = dt_raw.split('/')
                if len(dt_parts) != 3: continue
                
                try:
                    d, m, y = int(dt_parts[0]), int(dt_parts[1]), int(dt_parts[2])
                    merge_key = f"{y:04d}-{m:02d}-{d:02d}"
                except: continue
                
                # Parsing do valor: remove aspas, troca vírgula por ponto
                val_raw = row[1].strip().replace('"', '').replace("'", "").replace(',', '.')
                try:
                    r = float(val_raw)
                    if r > 1.0: r /= 100.0  # 14.5 -> 0.145
                    lookup[merge_key] = r
                except: continue
                
        if lookup:
            logger.info(f"  Selic carregada: {len(lookup)} datas de {total_linhas} linhas.")
            # Amostra da data mais recente no dicionário
            sample_keys = sorted(lookup.keys(), reverse=True)
            logger.info(f"  Selic Amostra (Recente): {sample_keys[0]} -> {lookup[sample_keys[0]]}")
        else:
            logger.error(f"  FALHA: Nenhuma data Selic parseada em {total_linhas} linhas!")
            
        return lookup
    except Exception as e:
        logger.error(f"Erro manual ao carregar Selic: {e}")
        return {}

def carregar_dividendos(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega Dividend Yields e sincroniza merge_key.
    """
    if not os.path.exists(caminho_arquivo):
        logger.warning(f"Arquivo de dividendos não encontrado: {caminho_arquivo}")
        return None
    try:
        df = pd.read_csv(caminho_arquivo, encoding='utf-8-sig')
        # Sincroniza timezone e data
        dt_col = 'data_only' if 'data_only' in df.columns else 'data'
        df['dt_obj'] = pd.to_datetime(df[dt_col], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
        df['merge_key'] = df['dt_obj'].dt.strftime('%Y-%m-%d').astype(str)
        
        # Padroniza ativo
        df['ativo'] = df['ativo'].astype(str).str.strip().str.extract('^([A-Z]{4})')[0]
        df['Dividend_Yield'] = df['Dividend_Yield'].fillna(0.0)
        
        return df[['merge_key', 'ativo', 'Dividend_Yield']].dropna(subset=['merge_key'])
    except Exception as e:
        logger.error(f"Erro ao ler Dividendos: {e}")
        return None

def carregar_ibov(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega IBOV e sincroniza merge_key.
    """
    if not os.path.exists(caminho_arquivo):
        logger.warning(f"Arquivo IBOV não encontrado: {caminho_arquivo}")
        return None
    try:
        df = pd.read_csv(caminho_arquivo, sep=';', decimal=',', skiprows=1, encoding='utf-8-sig')
        close_col = next((c for c in ['acao_close_ajustado', 'close', 'Close', 'CLOSE'] if c in df.columns), None)
        date_col = next((c for c in ['time', 'data', 'Data', 'date'] if c in df.columns), None)
        
        if not close_col or not date_col: return None
        
        df['dt_obj'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce').dt.normalize()
        df['merge_key'] = df['dt_obj'].dt.strftime('%Y-%m-%d').astype(str)
        
        df = df.sort_values('dt_obj')
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        df['log_ret_ibov'] = np.log(df[close_col] / df[close_col].shift(1)).fillna(0.0)
        
        return df[['merge_key', 'log_ret_ibov']].drop_duplicates('merge_key').dropna(subset=['merge_key'])
    except Exception as e:
        logger.error(f"Erro ao ler IBOV: {e}")
        return None

def calcular_peso_amostra(moneyness: float, price: float) -> float:
    """
    Calcula o peso da amostra para o treinamento.
    Foco: Opções baratas (OTM) e próximas ao dinheiro (ATM).
    """
    # 1. Ponderação por Moneyness (S/K para Call)
    if moneyness < 0.95:
        # Opções OTM: Precisam de peso alto para o modelo aprender o decaimento e o zero
        weight = 10.0
    elif 0.95 <= moneyness <= 1.05:
        # Opções ATM: Alta relevância para o mercado
        weight = 3.0
    else:
        # Opções ITM: Preços altos podem dominar o MSE, então reduzimos o peso relativo
        weight = 1.0
        
    # 2. Ponderação por Preço Absoluto (R$)
    # Se a opção for muito barata, aumentamos o peso para garantir precisão no centavo
    if price < 0.50:
        weight *= 2.0
        
    return weight

def _calcular_features_ativo(df_ativo: pd.DataFrame) -> pd.DataFrame:
    df = df_ativo.copy().sort_values('time')
    df['log_ret'] = np.log(df['spot_price'] / df['spot_price'].shift(1)).fillna(0.0)
    df['rolling_vol_20'] = df['log_ret'].rolling(window=20).std().fillna(0.0)
    df['ewma_vol'] = df['log_ret'].ewm(span=20, adjust=False).std().fillna(0.0)
    
    if 'acao_high' in df.columns and 'acao_low' in df.columns:
        h, l = pd.to_numeric(df['acao_high'], errors='coerce'), pd.to_numeric(df['acao_low'], errors='coerce')
        df['vol_parkinson'] = np.sqrt((1/(4*np.log(2))) * (np.log(h/(l+1e-8))**2)).fillna(0.0)
    else:
        df['vol_parkinson'] = 0.0
        
    if 'acao_vol_fin' in df.columns:
        df['log_vol_fin'] = np.log(pd.to_numeric(df['acao_vol_fin'], errors='coerce') + 1.0).fillna(0.0)
    else:
        df['log_vol_fin'] = 0.0
    return df

def criar_dataset_hibrido(caminho_pasta_opcoes: str, df_juros: pd.DataFrame, seq_length: int = 30, tickers: list = None):
    lista_dfs = []
    df_ibov = carregar_ibov(PATHS.get('ibov_data', os.path.join(os.path.dirname(caminho_pasta_opcoes), 'BOVA11.csv')))
    
    # Normalizar tickers para maiúsculas
    if tickers:
        tickers = [t.strip().upper() for t in tickers]

    for arquivo in os.listdir(caminho_pasta_opcoes):
        if not arquivo.endswith('.csv') or arquivo == "taxa_selic.csv": continue
        
        # Filtro por ticker
        if tickers:
            ticker_found = False
            for t in tickers:
                if t in arquivo.upper():
                    ticker_found = True
                    break
            if not ticker_found: continue

        try:
            path = os.path.join(caminho_pasta_opcoes, arquivo)
            df_temp = pd.read_csv(path, sep=';', decimal=',', skiprows=1, encoding='utf-8-sig')
            if 'time' not in df_temp.columns: continue
            
            # Parsing flexível de data para o ativo
            raw_time = df_temp['time'].astype(str).str.strip()
            # Tenta ISO primeiro, senão tenta formatos comuns
            df_temp['time_dt'] = pd.to_datetime(raw_time, errors='coerce')
            if df_temp['time_dt'].isna().mean() > 0.5:
                df_temp['time_dt'] = pd.to_datetime(raw_time, dayfirst=False, errors='coerce') # Tenta MM/DD
            
            # Sync merge_key e time
            df_temp['time'] = df_temp['time_dt'].dt.tz_localize(None)
            df_temp['merge_key'] = df_temp['time'].dt.normalize().dt.strftime('%Y-%m-%d').astype(str)
            df_temp['ativo'] = df_temp['symbol'].str[:4] if 'symbol' in df_temp.columns else 'UNKNOWN'
            
            if len(lista_dfs) == 0:
                logger.info(f"  Diagnóstico Ativo: raw='{raw_time.iloc[0]}' -> merge_key='{df_temp['merge_key'].iloc[0]}'")
            
            lista_dfs.append(df_temp)
        except: continue

    if not lista_dfs: return None, None
    df_full = pd.concat(lista_dfs, ignore_index=True)
    df_full = df_full[(df_full['time'] <= '2025-12-31') & (df_full['premium'] > 0)]
    
    logger.info(f"Data range: {df_full['time'].min()} a {df_full['time'].max()} | Registros: {len(df_full)}")
    
    # 1. Merge Selic (Via Lookup para evitar falhas de join do Pandas)
    if isinstance(df_juros, dict) and df_juros:
        df_full['r'] = df_full['merge_key'].map(df_juros)
        
        # Estatísticas de preenchimento
        nan_count = df_full['r'].isna().sum()
        if nan_count > 0:
            logger.warning(f"  Gap na Selic: {nan_count}/{len(df_full)} linhas sem match direto. Preenchendo...")
            # Detecta se foi falha total
            if nan_count == len(df_full):
                ex_key = df_full['merge_key'].iloc[0]
                logger.error(f"  FALHA CRÍTICA: Nenhuma data do dataset bateu com a Selic! Exemplo dataset: '{ex_key}'")
                if df_juros:
                    logger.info(f"  Chaves Selic disponíveis (5 primeiras): {list(df_juros.keys())[:5]}")
            
            # Ordena por tempo para propagar taxas próximas
            df_full = df_full.sort_values('time')
            df_full['r'] = df_full['r'].ffill().bfill().fillna(0.13)
        else:
            logger.info("  Selic vinculada com 100% de sucesso!")
    else:
        logger.warning("Dados Selic indisponíveis ou no formato errado. Usando fallback 0.13.")
        df_full['r'] = 0.13

    # 2. Merge Dividendos
    df_divs = carregar_dividendos(PATHS['dividend_data'])
    if df_divs is not None:
        # Cast explicitamente para string para evitar mismatch de tipos
        df_full['merge_key'] = df_full['merge_key'].astype(str)
        df_divs['merge_key'] = df_divs['merge_key'].astype(str)
        df_full['ativo'] = df_full['ativo'].astype(str)
        df_divs['ativo'] = df_divs['ativo'].astype(str)
        
        df_full = pd.merge(df_full, df_divs, on=['merge_key', 'ativo'], how='left')
        df_full['Dividend_Yield'] = df_full.groupby('ativo')['Dividend_Yield'].ffill().bfill().fillna(0.0)
    else:
        df_full['Dividend_Yield'] = 0.0

    # 3. Merge IBOV Global (para backup) e calculo por ativo
    salvar_dataset_integridade(df_full, "df_merged_treino_diagnostico.csv")
    
    ativos_unicos = sorted(df_full['ativo'].unique())
    asset_map = {ativo: i for i, ativo in enumerate(ativos_unicos)}
    
    sequences, pinn_inputs, targets, timestamps, asset_ids, sample_weights = [], [], [], [], [], []
    
    for nome_ativo, df_grupo in df_full.groupby('ativo'):
        df_grupo = df_grupo.sort_values('time')
        # Sub-dataset para features da LSTM
        df_asset = df_grupo[['time', 'spot_price', 'acao_high', 'acao_low', 'acao_vol_fin']].drop_duplicates('time')
        if len(df_asset) < seq_length + 20: continue
        
        df_asset = _calcular_features_ativo(df_asset)
        # Merge IBOV local para features
        if df_ibov is not None:
            df_asset['merge_key'] = df_asset['time'].dt.normalize().dt.strftime('%Y-%m-%d').astype(str)
            df_asset = pd.merge(df_asset, df_ibov, on='merge_key', how='left')
            df_asset['log_ret_ibov'] = df_asset['log_ret_ibov'].fillna(0.0)
            
        cols_lstm = ['log_ret', 'rolling_vol_20', 'ewma_vol', 'vol_parkinson', 'log_vol_fin', 'log_ret_ibov']
        asset_feats = df_asset[cols_lstm].values.astype(np.float32)
        asset_feats = np.nan_to_num(asset_feats)
        
        date_map = {ts: i for i, ts in enumerate(df_asset['time'].tolist())}
        
        for row in df_grupo.itertuples():
            idx = date_map.get(row.time)
            if idx is not None and idx >= seq_length:
                sequences.append(asset_feats[idx-seq_length : idx])
                pinn_inputs.append([row.spot_price, row.strike, row.days_to_maturity/252.0, row.r, row.Dividend_Yield])
                targets.append(row.premium)
                timestamps.append(row.time.timestamp())
                asset_ids.append(asset_map[nome_ativo])
                sample_weights.append(calcular_peso_amostra(row.spot_price/row.strike, row.premium))

    if not targets: return None, None
    
    X_seq = np.array(sequences, dtype=np.float32)
    X_phy = np.array(pinn_inputs, dtype=np.float32)
    y = np.array(targets, dtype=np.float32).reshape(-1, 1)
    
    # === STEP 0: Calculate y_norm BEFORE any normalization of X_phy ===
    # Using .copy() to ensure we don't use a view of normalized K
    K_raw = X_phy[:, 1:2].copy()
    y_norm = y / (K_raw + 1e-8)
    
    # Stats
    S, K, T, r, q = X_phy[:, 0], X_phy[:, 1], X_phy[:, 2], X_phy[:, 3], X_phy[:, 4]
    moneyness_arr = S / (K + 1e-8)
    data_stats = {
        'S_mean': float(S.mean()), 'S_std': float(S.std()), 'S_max': float(S.max()),
        'K_mean': float(K.mean()), 'K_std': float(K.std()), 'K_max': float(K.max()),
        'T_mean': float(T.mean()), 'T_std': float(T.std()), 'T_max': float(T.max()),
        'r_mean': float(r.mean()), 'r_std': float(r.std()), 'r_max': float(r.max()),
        'q_mean': float(q.mean()), 'q_std': float(q.std()), 'q_max': float(q.max()),
        'y_mean': float(y_norm.mean()),
        'y_std': float(y_norm.std()),
        'moneyness_mean': float(moneyness_arr.mean()),
        'asset_map': asset_map
    }
    
    # Global Z-Score for LSTM
    X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
    l_mean, l_std = X_seq_flat.mean(0), X_seq_flat.std(0) + 1e-8
    X_seq = (X_seq - l_mean) / l_std
    data_stats['lstm_feat_mean'], data_stats['lstm_feat_std'] = l_mean.tolist(), l_std.tolist()
    
    # Physics Normalization
    X_phy[:, 0] = (X_phy[:, 0] - data_stats['S_mean']) / (data_stats['S_std'] + 1e-8)
    X_phy[:, 1] = (X_phy[:, 1] - data_stats['K_mean']) / (data_stats['K_std'] + 1e-8)
    X_phy[:, 2] = (X_phy[:, 2] - data_stats['T_mean']) / (data_stats['T_std'] + 1e-8)
    X_phy[:, 3] = (X_phy[:, 3] - data_stats['r_mean']) / (data_stats['r_std'] + 1e-8)
    X_phy[:, 4] = (X_phy[:, 4] - data_stats['q_mean']) / (data_stats['q_std'] + 1e-8)
    
    # Weight normalization
    weights = np.array(sample_weights, dtype=np.float32).reshape(-1, 1)
    weights = weights / (weights.mean() + 1e-8)
    
    return TensorDataset(
        torch.from_numpy(X_seq), torch.from_numpy(X_phy), torch.from_numpy(y_norm.astype(np.float32)),
        torch.from_numpy(np.array(timestamps).reshape(-1,1)), 
        torch.from_numpy(weights),
        torch.from_numpy(np.array(asset_ids, dtype=np.int64))
    ), data_stats
