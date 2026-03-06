# /src/utils.py

import pandas as pd
import numpy as np
import torch
import os
from src.config import PATHS
from src.logger import get_logger

# Configurar logger
logger = get_logger('PINN_Utils')


def setup_results_dir():
    """
    Garante que a estrutura de diretórios para salvar resultados existe.
    """
    try:
        directories = [
            PATHS.get('results_dir'),
            PATHS.get('model_save_dir'),
            PATHS.get('plot_save_dir')
        ]
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Diretório de resultados preparado: {directory}")
    except Exception as e:
        logger.error(f"Erro ao configurar diretórios de resultados: {e}")


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
                    v = v + [None] * (max_len - len(v))
                history_normalized[k] = v

        df = pd.DataFrame(history_normalized)
        df.to_csv(save_path, index=False)
        logger.info(f"Histórico salvo com sucesso em: {save_path}")

    except Exception as e:
        logger.error(f"Erro crítico ao salvar histórico: {e}")
        try:
            pd.DataFrame.from_dict(history, orient='index').transpose().to_csv(save_path + ".bak")
            logger.info("Backup salvo.")
        except:
            pass


# ==============================================================================
# CALLBACK: VALIDAÇÃO DE NORMALIZAÇÃO PRÉ-TREINO
# ==============================================================================

def validate_normalization(
    train_loader,
    data_stats: dict,
    warn_threshold_mean: float = 0.5,
    warn_threshold_std: float = 0.5,
    critical_threshold_mean: float = 2.0,
    critical_std_min: float = 0.05,
    raise_on_critical: bool = True,
) -> str:
    """
    Callback de validação de normalização executado ANTES do treinamento.

    Verifica se X_phy (entradas da PINN) e X_seq (entradas da LSTM) estão
    normalizados em escala Z-Score (~mean=0, ~std=1). Emite warnings para
    desvios moderados e lança ValueError para violações críticas.

    Args:
        train_loader: DataLoader de treino (retorna batches com 6 elementos)
        data_stats: Dicionário de estatísticas do dataset (salvo pelo data_loader)
        warn_threshold_mean: Absoluto da média acima do qual emite WARNING (default: 0.5)
        warn_threshold_std: |std - 1| acima do qual emite WARNING (default: 0.5)
        critical_threshold_mean: Absoluto da média acima do qual é CRÍTICO (default: 2.0)
        critical_std_min: Std abaixo deste valor é CRÍTICO (feature constante) (default: 0.05)
        raise_on_critical: Se True, lança ValueError em violações críticas (default: True)

    Returns:
        String com o relatório completo de normalização.

    Raises:
        ValueError: Se raise_on_critical=True e violação crítica detectada.
    """
    # --- Coleta um batch de amostra ---
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        logger.error("DataLoader vazio! Não é possível validar normalização.")
        return "FALHA: DataLoader vazio."

    if len(batch) != 6:
        logger.error(f"Batch com {len(batch)} elementos (esperado 6). Validação abortada.")
        return f"FALHA: Batch inesperado ({len(batch)} elementos)."

    x_seq, x_phy, y_norm, _, weights, _ = batch

    # Converte para numpy para diagnóstico
    x_seq_np  = x_seq.float().numpy()   # (B, SEQ, FEAT)
    x_phy_np  = x_phy.float().numpy()   # (B, 5)
    y_np      = y_norm.float().numpy()  # (B, 1)
    w_np      = weights.float().numpy() # (B, 1)
    seq_flat  = x_seq_np.reshape(-1, x_seq_np.shape[-1])  # (B*SEQ, FEAT)

    issues_critical = []
    issues_warning  = []
    report_lines    = []

    # Definições de nomes
    phy_names = ['S (spot_norm)', 'K (strike_norm)', 'T (tau_norm)', 'r (rate_norm)', 'q (div_norm)']
    lstm_names = ['log_ret', 'rolling_vol_20', 'ewma_vol', 'vol_parkinson', 'log_vol_fin', 'log_ret_ibov']

    # --- Estatísticas do Target e Pesos ---
    y_flat = y_np.flatten()
    y_mean, y_std = float(np.mean(y_flat)), float(np.std(y_flat))
    y_min, y_max = float(np.min(y_flat)), float(np.max(y_flat))
    
    w_flat = w_np.flatten()
    w_mean, w_min = float(np.mean(w_flat)), float(np.min(w_flat))

    # --- Relatório Tabular (Modern Industrial) ---
    header = f"{'COMPONENTE':<12} | {'FEATURE':<20} | {'MEAN':>8} | {'STD':>7} | {'STATUS'}"
    sep = "-" * len(header)
    
    report_lines.append("")
    report_lines.append("=" * len(header))
    report_lines.append("           DIAGNÓSTICO DE NORMALIZAÇÃO PRÉ-TREINO")
    report_lines.append("=" * len(header))
    report_lines.append(header)
    report_lines.append(sep)

    def _format_status(mean, std, is_target=False, val_min=None, val_max=None):
        if is_target:
            if val_min < 0: return "[FAIL]"
            if val_max > 10.0: return "[WARN]"
            return "[PASS]"
        
        if abs(mean) > critical_threshold_mean or std < critical_std_min:
            return "[FAIL]"
        if abs(mean) > warn_threshold_mean or abs(std - 1.0) > warn_threshold_std:
            return "[WARN]"
        return "[PASS]"

    # 1. X_PHY
    for i, name in enumerate(phy_names):
        val = x_phy_np[:, i]
        m, s = float(np.mean(val)), float(np.std(val))
        status = _format_status(m, s)
        report_lines.append(f"{'PINN_INPUT':<12} | {name[:20]:<20} | {m:>8.4f} | {s:>7.4f} | {status}")
        if status == "[FAIL]": issues_critical.append(f"PINN_INPUT.{name}")
        if status == "[WARN]": issues_warning.append(f"PINN_INPUT.{name}")

    report_lines.append(sep)

    # 2. X_SEQ
    for i in range(seq_flat.shape[1]):
        name = lstm_names[i] if i < len(lstm_names) else f'feat_{i}'
        val = seq_flat[:, i]
        m, s = float(np.mean(val)), float(np.std(val))
        status = _format_status(m, s)
        report_lines.append(f"{'LSTM_SEQ':<12} | {name[:20]:<20} | {m:>8.4f} | {s:>7.4f} | {status}")
        if status == "[FAIL]": issues_critical.append(f"LSTM_SEQ.{name}")
        if status == "[WARN]": issues_warning.append(f"LSTM_SEQ.{name}")

    report_lines.append(sep)

    # 3. Target e Pesos
    y_status = _format_status(y_mean, y_std, is_target=True, val_min=y_min, val_max=y_max)
    report_lines.append(f"{'TARGET':<12} | {'y_norm (P/K)':<20} | {y_mean:>8.4f} | {y_std:>7.4f} | {y_status}")
    
    w_status = "[PASS]" if w_min >= 0 and abs(w_mean-1) < 0.3 else "[WARN]"
    report_lines.append(f"{'WEIGHTS':<12} | {'sampling':<20} | {w_mean:>8.4f} | {'-':>7} | {w_status}")

    report_lines.append(sep)

    # 4. Consistência data_stats
    expected_keys = ['S_mean', 'S_std', 'K_mean', 'K_std', 'T_mean', 'T_std', 'r_mean', 'r_std', 'q_mean', 'q_std']
    missing = [k for k in expected_keys if k not in data_stats]
    if missing or 'lstm_feat_mean' not in data_stats:
        report_lines.append(f"{'SYSTEM':<12} | {'data_stats':<20} | {'-':>8} | {'-':>7} | [WARN]")
        issues_warning.append("data_stats.missing_keys")
    else:
        report_lines.append(f"{'SYSTEM':<12} | {'data_stats':<20} | {'-':>8} | {'-':>7} | [PASS]")

    report_lines.append("=" * len(header))
    
    # 5. Sumário
    if issues_critical:
        report_lines.append(f" RESULTADO: {len(issues_critical)} FALHA(S) CRÍTICA(S) DETECTADA(S)")
    elif issues_warning:
        report_lines.append(f" RESULTADO: {len(issues_warning)} ANOMALIA(S) DETECTADA(S)")
    else:
        report_lines.append(" RESULTADO: SUCESSO - Dados validados para treinamento")
    report_lines.append("=" * len(header))

    # Loga o relatório completo (usando o logger interno para evitar poluição no main)
    for line in report_lines:
        logger.info(line)
    
    full_report = "\n".join(report_lines)

    # Lança exceção em caso de violações críticas
    if issues_critical and raise_on_critical:
        raise ValueError(
            f"Erro de Normalização: {len(issues_critical)} falhas críticas.\n"
            "Verifique os logs acima para detalhes."
        )

    return full_report

    full_report = "\n".join(report_lines)

    # Loga o relatório completo
    for line in report_lines:
        if "CRÍTICO" in line or "🔴" in line:
            logger.error(line)
        elif "WARN" in line or "⚠️" in line:
            logger.warning(line)
        else:
            logger.info(line)

    # Lança exceção em caso de violações críticas
    if issues_critical and raise_on_critical:
        raise ValueError(
            f"[validate_normalization] {len(issues_critical)} violação(ões) crítica(s) "
            f"detectada(s). Corrija os dados antes de treinar.\n"
            + "\n".join(issues_critical)
        )

    return full_report

def salvar_dataset_integridade(df: pd.DataFrame, filename: str):
    """
    Salva o dataframe resultante do merge para análise de integridade.
    """
    try:
        results_dir = PATHS['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, filename)
        df.to_csv(save_path, index=False)
        logger.info(f"Dataset de integridade salvo em: {save_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar dataset de integridade: {e}")
