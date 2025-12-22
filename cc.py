# /src/visualization.py

"""
Módulo de Visualização e Diagnóstico (PINN Heston).

Responsável por gerar artefatos visuais para validação do modelo híbrido:
1. Curvas de aprendizado (Loss History).
2. Comparativos de Preço (Real vs Modelo vs Black-Scholes).
3. Superfícies 3D (Preço, Delta, Resíduo PDE).
4. Análise de Erro (Distribuição, Moneyness, Heatmaps).
5. Dinâmica dos Parâmetros Estocásticos (Heston).

Padrão: Seaborn/Matplotlib para relatórios financeiros.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import r2_score

from src.model import DeepHestonHybrid
from src.physics import heston_residual
from src.config import VIZ_CONFIG, PATHS
from src.logger import get_logger

# Configurar logger
logger = get_logger('PINN_Visualization')

# Estilo profissional para publicações financeiras
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10, 
    'axes.titlesize': 12, 
    'axes.labelsize': 10, 
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})

# ==============================================================================
# FUNÇÕES UTILITÁRIAS (Black-Scholes e Vol Implícita)
# ==============================================================================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Cálculo analítico exato de Black-Scholes.
    Usado como baseline para comparação de erro do modelo PINN.
    """
    # Proteção numérica
    S = np.maximum(S, 1e-5)
    K = np.maximum(K, 1e-5)
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price

def bs_implied_vol_approx(S, K, T, r, market_price, option_type='call'):
    """
    Aproximação rápida da Volatilidade Implícita (Brenner & Subrahmanyam).
    Útil para plotar o 'Smile' sem resolver a inversa numericamente ponto a ponto.
    """
    # Filtra valores inválidos para evitar log(negativo)
    valid = (market_price > 0) & (T > 0)
    sigma = np.full_like(market_price, np.nan)
    
    if np.any(valid):
        # Aproximação simples para ATM
        sigma[valid] = (market_price[valid] / S[valid]) * np.sqrt(2 * np.pi / T[valid])
    
    return sigma

# ==============================================================================
# CLASSE VISUALIZER
# ==============================================================================

class Visualizer:
    """
    Gerenciador central de visualizações do Pipeline.
    Recebe o modelo treinado e os dados de validação para gerar relatórios.
    """

    def __init__(self, model, history_path, val_loader, data_stats, config):
        """
        Inicializa o visualizador com o contexto necessário.

        Args:
            model (nn.Module): O modelo híbrido (LSTM+PINN) já treinado.
            history_path (str): Caminho para o CSV com o log de perdas (Losses).
            val_loader (DataLoader): Dados de validação para inferência (Backtest).
            data_stats (dict): Estatísticas de normalização (Mean/Std) para desnormalizar plots.
            config (dict): Dicionário VIZ_CONFIG controlando quais plots gerar.
        """
        self.model = model
        self.history_path = history_path
        self.val_loader = val_loader
        self.data_stats = data_stats
        self.config = config
        self.device = next(model.parameters()).device # Detecta device do modelo
        
        # Cria diretório de salvamento se não existir
        os.makedirs(PATHS['plot_save_dir'], exist_ok=True)
        
        # Diretório de salvamento
        self.save_dir = self.config.get('plot_save_dir', PATHS['plot_save_dir'])
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Cache para inferência (evita reprocessar forward pass múltiplos vezes)
        self.preds_cache = None

    def _run_inference(self):
        """
        Roda inferência em todo o dataset de validação e faz cache dos resultados.
        Essencial para performance, evitando múltiplos loops no DataLoader.
        """
        if self.preds_cache is not None: return self.preds_cache

        self.model.eval()
        results = {'pred': [], 'real': [], 'params': [], 'phy': [], 'times': []}

        logger.info("Executando inferência completa no dataset de validação...")
        with torch.no_grad():
            for batch in self.val_loader:
                x_seq = batch[0].to(self.device)
                x_phy = batch[1].to(self.device)
                y_real = batch[2].to(self.device)
                asset_ids = batch[5].to(self.device) # ID real

                outputs = self.model(x_seq, x_phy, asset_ids)
                
                results['pred'].append(outputs['price'].cpu().numpy())
                results['real'].append(y_real.cpu().numpy())
                results['params'].append(torch.cat(outputs['heston_params'], dim=1).cpu().numpy())
                results['phy'].append(x_phy.cpu().numpy())
                
                if len(batch) > 3:
                    results['times'].append(batch[3].numpy())

        # Concatenação segura
        self.preds_cache = {
            'price_pred': np.concatenate(results['pred']),
            'price_real': np.concatenate(results['real']),
            'heston_params': np.concatenate(results['params']),
            'inputs_phy': np.concatenate(results['phy']),
            'times': np.concatenate(results['times']) if results['times'] else None
        }
        return self.preds_cache

    def _generate_synthetic_grid(self, n_points=40):
        """
        Gera grid (S, T) sintético para plots de superfície e derivadas.
        Permite visualizar a 'suavidade' da solução aprendida pela PINN.
        """
        # Pega uma amostra real para gerar o estado inicial da LSTM (Market State)
        try:
            sample_batch = next(iter(self.val_loader))
        except StopIteration:
            logger.error("Dataloader vazio. Não é possível gerar grid sintético.")
            return None, None, None, None, None

        x_seq_sample = sample_batch[0].to(self.device)
        asset_ids_sample = sample_batch[5].to(self.device)
        
        with torch.no_grad():
            # CORREÇÃO: Preparar input da LSTM com embeddings
            if self.model.use_embedding:
                # [Batch, Emb_Dim]
                emb = self.model.asset_embedding(asset_ids_sample)
                # [Batch, Seq, Emb_Dim]
                emb_seq = emb.unsqueeze(1).repeat(1, x_seq_sample.size(1), 1)
                # [Batch, Seq, Features+Emb]
                lstm_input = torch.cat([x_seq_sample, emb_seq], dim=2)
            else:
                lstm_input = x_seq_sample

            _, (h_n, _) = self.model.lstm(lstm_input)
            
            # Tira a média do estado de mercado para ter um "regime médio"
            market_state_avg = h_n[-1].mean(dim=0, keepdim=True)
            market_state_expanded = market_state_avg.repeat(n_points*n_points, 1)
            # Gera parâmetros Heston baseados nesse regime médio
            nu, theta, kappa, xi, rho = self.model.heston_head(market_state_expanded)

        # Grid físico
        s_vals = np.linspace(0.6, 1.4, n_points) # Moneyness range
        t_vals = np.linspace(0.1, 1.0, n_points) # Time range (anos)
        S_grid, T_grid = np.meshgrid(s_vals, t_vals)
        
        # Converte para tensores [N*N, 1] - garantindo device correto
        S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=self.device).unsqueeze(1)
        T_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=self.device).unsqueeze(1)
        K_flat = torch.ones_like(S_flat, device=self.device) * 1.0   # Strike normalizado
        r_flat = torch.ones_like(S_flat, device=self.device) * 0.10  # Juros fixos 10%
        q_flat = torch.ones_like(S_flat, device=self.device) * 0.02  # Dividend yield fixo 2%
        
        # Input físico: [S, K, T, r, q] - forçar device explicitamente
        x_phy_synthetic = torch.cat([S_flat, K_flat, T_flat, r_flat, q_flat], dim=1).to(self.device).requires_grad_(True)        
        
        # Asset ID Fictício (Usa 0 como padrão para o plot genérico)
        asset_ids_synthetic = torch.zeros(x_phy_synthetic.shape[0], dtype=torch.long, device=self.device)

        # Parâmetros Heston (com gradiente habilitado para a EDP)
        # Detach para quebrar o grafo da LSTM (não precisamos derivar até a LSTM aqui)
        # Garantir que permaneçam no device correto
        fixed_heston_params = (
            nu.detach().to(self.device).requires_grad_(True),
            theta.detach().to(self.device).requires_grad_(True),
            kappa.detach().to(self.device).requires_grad_(True),
            xi.detach().to(self.device).requires_grad_(True),
            rho.detach().to(self.device).requires_grad_(True)
        )
        
        return x_phy_synthetic, fixed_heston_params, S_grid, T_grid, asset_ids_synthetic
    
    def _safe_reshape_grid(self, array, grid_shape):
        """Helper para evitar o crash 'cannot reshape array of size 1'."""
        if array.size == 1:
            logger.warning("Resíduo/Erro escalar detectado. Fazendo broadcast para o grid inteiro.")
            return np.full(grid_shape, array.item())
        return array.reshape(grid_shape)

    # =========================================================================
    # PLOTS DE DIAGNÓSTICO E PERFORMANCE
    # =========================================================================

    def plot_prediction_scatter(self):
        """
        Gráfico Scatter: Previsão (Y) vs Real (X).
        Colorido pelo Tempo para Maturidade (T) para identificar viés temporal.
        """
        try:
            data = self._run_inference()
            y_real = data['price_real'].flatten()
            y_pred = data['price_pred'].flatten()
            t_maturity = data['inputs_phy'][:, 2].flatten()
            
            r2 = r2_score(y_real, y_pred)
            
            plt.figure(figsize=(7, 6))
            # Otimização: se tiver muitos pontos, plota sem borda e menor
            s_size = 5 if len(y_real) > 10000 else 10
            sc = plt.scatter(y_real, y_pred, alpha=0.3, s=s_size, c=t_maturity, cmap='viridis', label='Amostras')
            
            max_val = max(y_real.max(), y_pred.max())
            plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Identidade')
            
            plt.colorbar(sc, label='Tempo para Vencimento (Norm)')
            plt.title(f'Previsão vs Real (R² = {r2:.4f})')
            plt.xlabel('Preço Real')
            plt.ylabel('Preço Previsto')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'prediction_scatter.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Erro no scatter plot: {e}")

    def plot_premium_over_time(self):
        """
        Série Temporal: Média diária de preços (Real vs Previsto).
        Verifica se o modelo captura a tendência macro do mercado.
        """
        try:
            data = self._run_inference()
            y_real, y_pred = data['price_real'].flatten(), data['price_pred'].flatten()
            if data['times'] is not None:
                df = pd.DataFrame({'timestamp': data['times'].flatten(), 'real': y_real, 'pred': y_pred})
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                df_plot = df.sort_values('date').groupby(df['date'].dt.date)[['real', 'pred']].mean()
                plt.figure(figsize=(10, 5))
                plt.plot(df_plot.index, df_plot['real'], label='Mercado', color='black', alpha=0.6)
                plt.plot(df_plot.index, df_plot['pred'], label='Modelo', color='red', linestyle='--', alpha=0.8)
                plt.title('Média Diária de Preços (Global)')
                plt.xlabel('Data'); plt.ylabel('Preço Médio'); plt.legend(); plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45); plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, 'premium_over_time.png')); plt.close()
        except Exception as e: logger.error(f"Erro premium_over_time: {e}")

    def plot_premium_by_moneyness_time(self):
        """
        Série Temporal Segmentada: ITM, ATM, OTM.
        Permite identificar se o erro está concentrado em alguma região específica.
        """
        try:
            data = self._run_inference()
            if data['times'] is None: return
            
            S = data['inputs_phy'][:, 0].flatten() * (self.data_stats['S_max'] - self.data_stats['S_min']) + self.data_stats['S_min']
            K = data['inputs_phy'][:, 1].flatten() * (self.data_stats['K_max'] - self.data_stats['K_min']) + self.data_stats['K_min']
            moneyness = S / K
            
            df = pd.DataFrame({
                'date': pd.to_datetime(data['times'].flatten(), unit='s'),
                'real': data['price_real'].flatten(),
                'pred': data['price_pred'].flatten(),
                'moneyness': moneyness
            })
            df['cat'] = pd.cut(df['moneyness'], [0, 0.97, 1.03, np.inf], labels=['OTM', 'ATM', 'ITM'])
            
            # Agrupa por data e categoria
            df_grp = df.groupby([df['date'].dt.date, 'cat'], observed=True)[['real', 'pred']].mean().reset_index()
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            cats = ['ITM', 'ATM', 'OTM']
            colors = ['green', 'blue', 'orange']
            
            for i, cat in enumerate(cats):
                subset = df_grp[df_grp['cat'] == cat]
                ax = axes[i]
                ax.plot(subset['date'], subset['real'], label=f'Real {cat}', color='black', alpha=0.5)
                ax.plot(subset['date'], subset['pred'], label=f'Pred {cat}', color=colors[i], linestyle='--')
                ax.set_title(f'Evolução Temporal: {cat}')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
            
            plt.xlabel('Data')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'premium_by_moneyness_time.png'))
            plt.close()
        except Exception as e: logger.error(f"Erro premium_by_moneyness_time: {e}")

    def plot_distribution_overlay(self):
        """
        Histograma Comparativo: Densidade Real vs Prevista.
        Detecta colapso de modo (ex: se o modelo prevê sempre a média).
        """
        try:
            data = self._run_inference()
            y_real = data['price_real'].flatten()
            y_pred = data['price_pred'].flatten()

            plt.figure(figsize=(8, 5))
            sns.kdeplot(y_real, fill=True, label='Real (Mercado)', color='blue', alpha=0.3)
            sns.kdeplot(y_pred, fill=True, label='Modelo (Previsto)', color='red', alpha=0.3)
            
            plt.title('Sobreposição de Distribuição de Preços')
            plt.xlabel('Preço da Opção')
            plt.ylabel('Densidade')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'distribution_overlay.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Erro no distribution overlay: {e}")

    def plot_residuals(self):
        """
        Scatter: Resíduo (Erro) vs Moneyness.
        Verifica se há viés sistemático (ex: erro cresce ITM?).
        """
        try:
            data = self._run_inference()
            y_real = data['price_real'].flatten()
            y_pred = data['price_pred'].flatten()
            
            S = data['inputs_phy'][:, 0].flatten()
            K = data['inputs_phy'][:, 1].flatten()
            moneyness = S / (K + 1e-8) 
            
            residuals = y_pred - y_real
            
            plt.figure(figsize=(8, 5))
            plt.scatter(moneyness, residuals, alpha=0.2, s=10, color='purple')
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            
            plt.title('Análise de Resíduos (Viés)')
            plt.xlabel('Moneyness (S/K)')
            plt.ylabel('Resíduo (Pred - Real)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'residuals_analysis.png'))
            plt.close()
        except Exception as e:
             logger.error(f"Erro no plot de resíduos: {e}")

    def plot_error_by_moneyness(self):
        """
        Barplot: Erro Médio Absoluto (MAE) por Região.
        Quantifica onde o modelo erra mais.
        """
        try:
            data = self._run_inference()
            S = data['inputs_phy'][:, 0].flatten() * (self.data_stats['S_max'] - self.data_stats['S_min']) + self.data_stats['S_min']
            K = data['inputs_phy'][:, 1].flatten() * (self.data_stats['K_max'] - self.data_stats['K_min']) + self.data_stats['K_min']
            
            err = np.abs(data['price_pred'].flatten() - data['price_real'].flatten())
            df = pd.DataFrame({'moneyness': S/K, 'error': err})
            df['cat'] = pd.cut(df['moneyness'], [0, 0.95, 1.05, np.inf], labels=['OTM', 'ATM', 'ITM'])
            
            err_mean = df.groupby('cat', observed=True)['error'].mean()
            plt.figure(figsize=(6, 4))
            err_mean.plot(kind='bar', color=['salmon', 'lightblue', 'lightgreen'], edgecolor='black', alpha=0.8)
            plt.title('Erro Médio Absoluto (MAE) por Moneyness')
            plt.ylabel('MAE')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'error_by_moneyness.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"Não foi possível plotar erro por moneyness: {e}")

    def plot_error_distribution(self):
        """
        Diagnóstico de Normalidade dos Resíduos.
        Inclui Histograma e QQ-Plot.
        """
        try:
            from scipy import stats
            
            data = self._run_inference()
            y_real = data['price_real'].flatten()
            y_pred = data['price_pred'].flatten()
            
            errors = y_pred - y_real
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histograma dos erros com KDE
            ax1 = axes[0]
            sns.histplot(errors, bins=50, kde=True, ax=ax1, color='skyblue', edgecolor='black', alpha=0.7)
            ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
            ax1.set_xlabel('Erro (Previsto - Real)')
            ax1.set_ylabel('Frequência')
            ax1.set_title('Distribuição dos Erros de Previsão')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Estatísticas
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax1.text(0.02, 0.98, f'Média: {mean_error:.4f}\nStd: {std_error:.4f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # QQ-plot dos resíduos
            ax2 = axes[1]
            # Limitar a 5000 pontos para performance
            sample_size = min(5000, len(errors))
            errors_sample = np.random.choice(errors, sample_size, replace=False)
            stats.probplot(errors_sample, dist="norm", plot=ax2)
            ax2.set_title('QQ-Plot dos Resíduos (Normalidade)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self