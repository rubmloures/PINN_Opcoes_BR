# /src/visualization.py

"""
Módulo de Visualização e Diagnóstico Integrado (PINN Heston + LSTM).

Este módulo consolida todas as métricas e diagnósticos do pipeline:
1. Validação do LSTM (vs Ground Truth via calibração inversa)
2. Validação da PINN (Convergência e Física)
3. Validação do Modelo Integrado (Precificação de Mercado)

Padrão Gráfico: Plotly (interativo, versátil para dashboards)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

from src.config import VIZ_CONFIG, PATHS, DATA_CONFIG
from src.physics import heston_residual, PhysicsUtils
from src.logger import get_logger

# Configurar logger exclusivo para o módulo visual
logger = get_logger('PINN_Visualization')

# --- Configuração Global de Estilo ---
# Define um estilo limpo e profissional, ideal para papers ou dashboards.
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10, 
    'axes.titlesize': 12, 
    'axes.labelsize': 10, 
    'figure.dpi': 150,      # Alta resolução para exportação
    'savefig.bbox': 'tight' # Evita cortes nas legendas ao salvar
})

# ==============================================================================
# SEÇÃO 1: Funções Utilitárias Financeiras
# ==============================================================================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calcula o preço teórico de uma opção Europeia via Black-Scholes-Merton.
    Usado aqui apenas como 'Baseline' para comparação de erro.

    Args:
        S (float/array): Preço Spot do ativo.
        K (float/array): Preço de Exercício (Strike).
        T (float/array): Tempo até o vencimento (em anos).
        r (float/array): Taxa livre de risco anualizada.
        sigma (float/array): Volatilidade implícita ou histórica.
        option_type (str): 'call' ou 'put'.

    Returns:
        np.array: Preço teórico da opção.
    """
    # Proteção numérica para evitar divisão por zero ou log(0)
    S = np.maximum(S, 1e-5)
    K = np.maximum(K, 1e-5)
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)
    
    # Cálculo das probabilidades d1 e d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price

def bs_implied_vol_approx(S, K, T, r, price):
    """Aproximação Brenner & Subrahmanyam para Vol Implícita."""
    T = np.maximum(T, 1e-5)
    S = np.maximum(S, 1e-5)
    price = np.maximum(price, 1e-5)
    return np.sqrt(2 * np.pi / T.flatten()) * (price.flatten() / S.flatten())

# ==============================================================================
# SEÇÃO 2: Classe Visualizer (Motor de Plotagem)
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
        # Pega uma amostra real para gerar o estado inicial da LSTM
        try:
            sample_batch = next(iter(self.val_loader))
        except StopIteration:
            logger.error("Dataloader vazio. Não é possível gerar grid sintético.")
            return None, None, None, None, None

        x_seq_sample = sample_batch[0].to(self.device)
        asset_ids_sample = sample_batch[5].to(self.device)
        
        with torch.no_grad():
            #Preparar input da LSTM com embeddings
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
            # 1. Recuperar K Real para desnormalizar preços
            K_norm = data['inputs_phy'][:, 1]
            K_real = self._denormalize(K_norm, 'K')
            
            # 2. Preços Reais (P = P_norm * K_real)
            y_real = data['price_real'].flatten() * K_real
            y_pred = data['price_pred'].flatten() * K_real
            
            # 3. Tempo Real (para cor)
            T_real = self._denormalize(data['inputs_phy'][:, 2], 'T').flatten()
            
            r2 = r2_score(y_real, y_pred)
            
            plt.figure(figsize=(7, 6))
            # Otimização: se tiver muitos pontos, plota sem borda e menor
            s_size = 5 if len(y_real) > 10000 else 10
            sc = plt.scatter(y_real, y_pred, alpha=0.3, s=s_size, c=T_real, cmap='viridis', label='Amostras')
            
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
            
            # 1. Desnormalizar Inputs
            S_real = self._denormalize(data['inputs_phy'][:, 0], 'S').flatten()
            K_real = self._denormalize(data['inputs_phy'][:, 1], 'K').flatten()
            
            # 2. Desnormalizar Preços
            y_real = data['price_real'].flatten() * K_real
            y_pred = data['price_pred'].flatten() * K_real
            
            # 3. Calcular Moneyness Real (S/K) e Resíduos
            moneyness = S_real / (K_real + 1e-8)
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
            plt.savefig(os.path.join(self.save_dir, 'error_distribution.png'))
            plt.close()
            
            logger.info(f"Distribuição de erros plotada. Média={mean_error:.4f}, Std={std_error:.4f}")
            
        except Exception as e:
            logger.error(f"Erro ao plotar distribuição de erros: {e}")

    def plot_heston_params(self):
        """
        Distribuição dos Parâmetros Heston Inferidos pela PINN, 
        para verificar se estão em faixas realistas. 
        """
        try:
            data = self._run_inference()
            params = data['heston_params']
            names = [r'$\nu_0$ (Var)', r'$\theta$ (Long Var)', r'$\kappa$ (Rev)', r'$\xi$ (VolVol)', r'$\rho$ (Corr)']
            
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i, ax in enumerate(axes):
                sns.histplot(params[:, i], kde=True, ax=ax, color='skyblue', edgecolor='none')
                ax.set_title(names[i])
                ax.set_xlabel('')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'heston_params_dist.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Erro ao plotar params: {e}")

    def plot_latent_vol_evolution(self):
        """
        Dinâmica da Volatilidade Estocástica ao Longo do Tempo.
        """
        try:
            data = self._run_inference()
            nu_pred = data['heston_params'][:, 0]
            limit = min(2000, len(nu_pred))
            
            plt.figure(figsize=(10, 4))
            plt.plot(nu_pred[-limit:], color='darkorange', linewidth=1)
            plt.title(f'Dinâmica da Variância Estocástica (Últimos {limit} pontos)')
            plt.ylabel(r'Variância ($\nu_t$)')
            plt.xlabel('Tempo (Amostras)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'latent_vol_evolution.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Erro ao plotar vol evolution: {e}")

    def plot_delta_surface(self):
        """
        Superfície do Delta (dV/dS) Aprendido pela PINN. 
        """
        try:
            x_phy, heston_params, S_grid, T_grid, asset_ids = self._generate_synthetic_grid()
            if x_phy is None: return

            nu, theta, kappa, xi, rho = heston_params
            pinn_input = torch.cat([x_phy, nu, theta, kappa, xi, rho], dim=1)
            feats = self.model.fourier_layer(pinn_input) if self.model.use_fourier else pinn_input
            price = torch.nn.functional.softplus(self.model.pricing_net(feats))
            
            # Gradiente dV/dS (S está no índice 0 de x_phy)
            grads = torch.autograd.grad(price, x_phy, torch.ones_like(price), create_graph=False)[0]
            delta = grads[:, 0].cpu().numpy()
            delta_grid = self._safe_reshape_grid(delta, S_grid.shape)
            
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(S_grid, T_grid, delta_grid, cmap='coolwarm', edgecolor='none')
            ax.set_title('Superfície do Delta (Aprendido)')
            ax.set_xlabel('Spot')
            ax.set_ylabel('Tempo')
            ax.set_zlabel('Delta')
            fig.colorbar(surf, shrink=0.5)
            plt.savefig(os.path.join(self.save_dir, 'delta_surface.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Erro ao gerar delta surface: {e}")

    def plot_model_vs_bs_comparison(self):
        """
        Comparação com Black-Scholes.
        Plota Preço Real vs Modelo vs BS com Volatilidade Fixa.
        """
        try:
            data = self._run_inference()
            idx = np.random.choice(len(data['price_pred']), min(5000, len(data['price_pred'])), replace=False)
            
            # 1. Desnormalização Obrigatória
            S = self._denormalize(data['inputs_phy'][idx, 0], 'S')
            K = self._denormalize(data['inputs_phy'][idx, 1], 'K')
            T = self._denormalize(data['inputs_phy'][idx, 2], 'T')
            r = self._denormalize(data['inputs_phy'][idx, 3], 'r')
            
            real = data['price_real'][idx] * K
            pred = data['price_pred'][idx] * K
            
            bs = black_scholes_price(S, K, T, r, 0.30)
            moneyness = S/K
            mask = (T > 0.1) & (T < 0.5)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(moneyness[mask], real[mask], c='black', alpha=0.2, s=15, label='Real')
            plt.scatter(moneyness[mask], bs[mask], c='blue', alpha=0.1, s=15, label='BS (Flat Vol)')
            plt.scatter(moneyness[mask], pred[mask], c='red', alpha=0.2, s=15, label='Modelo')
            
            plt.xlabel('Moneyness (S/K)')
            plt.ylabel('Preço')
            plt.title('Comparativo de Precificação')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'model_vs_bs_comparison.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Erro ao comparar BS: {e}")

    def plot_vol_smile(self):
        """
        Plot Smile de Volatilidade Implícita.
        """
        try:
            data = self._run_inference()
            # Amostra aleatória para não pesar
            idx = np.random.choice(len(data['price_pred']), min(8000, len(data['price_pred'])), replace=False)
            
            # 1. Desnormalização Completa
            S = self._denormalize(data['inputs_phy'][idx, 0], 'S')
            K = self._denormalize(data['inputs_phy'][idx, 1], 'K')
            T = self._denormalize(data['inputs_phy'][idx, 2], 'T')
            r = self._denormalize(data['inputs_phy'][idx, 3], 'r')
            
            # Preço previsto em Reais
            prices = data['price_pred'][idx] * K
            
            # Filtro mais permissivo: Maturidade entre 2 semanas e 6 meses
            mask = (T > 0.04) & (T < 0.5) 
            if mask.sum() < 50: return

            iv = bs_implied_vol_approx(S[mask], K[mask], T[mask], r[mask], prices[mask])
            mon = K[mask] / S[mask] # Moneyness do Smile (K/S)
            t_vals = T[mask] # Para cor
            
            # Limpeza de NaNs/Infs
            valid_idx = np.isfinite(iv) & np.isfinite(mon) & np.isfinite(t_vals)
            if valid_idx.sum() < 10:
                logger.warning("Poucos dados válidos para Vol Smile após limpeza.")
                return

            mon = mon[valid_idx]
            iv = iv[valid_idx]
            t_vals = t_vals[valid_idx]
            
            plt.figure(figsize=(9, 6))
            # Scatter colorido pelo tempo de vencimento
            sc = plt.scatter(mon, iv, alpha=0.6, c=t_vals, cmap='plasma', s=15)
            
            # Linhas de tendência quadrática para curto e médio prazo
            try:
                # Tendência Curto prazo
                mask_short = t_vals < 0.15
                if mask_short.sum() > 10:
                    p_short = np.poly1d(np.polyfit(mon[mask_short], iv[mask_short], 2))
                    x_r = np.linspace(mon.min(), mon.max(), 100)
                    plt.plot(x_r, p_short(x_r), 'k--', label='Tendência (Curto Prazo)', linewidth=1.5)
            except: pass

            plt.colorbar(sc, label='Tempo para Vencimento (Anos)')
            plt.title('Smile de Volatilidade Implícita (Colorido por Maturidade)')
            plt.xlabel('Moneyness (K/S)')
            plt.ylabel('IV Anualizada')
            plt.legend()
            plt.grid(True, alpha=0.3)
            try:
                plt.savefig(os.path.join(self.save_dir, 'volatility_smile.png'))
            except OSError as e:
                logger.error(f"Falha de I/O ao salvar vol_smile: {e}")
            plt.close()
        except Exception as e: logger.error(f"Erro no vol smile: {e}")

    def plot_price_surface(self):
        """Superfície 3D de Preço."""
        try:
            x_phy, heston_params, S_grid, T_grid, asset_ids = self._generate_synthetic_grid()
            if x_phy is None: return

            nu, theta, kappa, xi, rho = heston_params
            pinn_input = torch.cat([x_phy, nu, theta, kappa, xi, rho], dim=1)
            
            # Projeção Fourier
            feats = self.model.fourier_layer(pinn_input) if self.model.use_fourier else pinn_input
            
            with torch.no_grad(): 
                nn_out = self.model.pricing_net(feats)
                # Aplicar Hard Constraint manualmente para visualização correta
                T_norm = x_phy[:, 2:3]
                time_factor = 1.0 - torch.exp(-10.0 * T_norm)
                time_value = time_factor * torch.nn.functional.softplus(nn_out)
                
                # Payoff (S e K estão normalizados e alinhados no grid)
                payoff = torch.relu(x_phy[:, 0:1] - x_phy[:, 1:2])
                price = payoff + time_value
            
            Z = self._safe_reshape_grid(price.cpu().numpy(), S_grid.shape)
            
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(S_grid, T_grid, Z, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel('Spot (Moneyness)')
            ax.set_ylabel('Tempo (Anos)')
            ax.set_zlabel('Preço Normalizado')
            ax.set_title('Superfície de Preço Aprendida (Híbrida)')
            fig.colorbar(surf, shrink=0.5, aspect=10)
            plt.savefig(os.path.join(self.save_dir, 'price_surface_3d.png'))
            plt.close()
        except Exception as e: 
            logger.error(f"Erro price_surface: {e}")

    def plot_error_heatmap(self):
        """(NOVO) Heatmap 2D do Erro Físico."""
        try:
            x_phy, heston_params, S_grid, T_grid, asset_ids = self._generate_synthetic_grid(n_points=50)
            if x_phy is None: return

            nu, theta, kappa, xi, rho = heston_params
            pinn_input = torch.cat([x_phy, nu, theta, kappa, xi, rho], dim=1)
            feats = self.model.fourier_layer(pinn_input) if self.model.use_fourier else pinn_input
            price = torch.nn.functional.softplus(self.model.pricing_net(feats))
            
            # Calcula Resíduo Ponto-a-Ponto
            model_output = {'price': price, 'heston_params': heston_params}
            res = heston_residual(model_output, x_phy, self.data_stats, return_residuals=True)
            
            res_abs = torch.abs(res).detach().cpu().numpy()
            res_grid = self._safe_reshape_grid(res_abs, S_grid.shape)
            
            plt.figure(figsize=(8, 6))
            plt.contourf(S_grid, T_grid, res_grid, levels=50, cmap='inferno')
            plt.colorbar(label='|Resíduo PDE|')
            plt.xlabel('Spot Price (Normalizado)')
            plt.ylabel('Tempo até Vencimento')
            plt.title('Mapa de Calor do Erro Físico')
            plt.savefig(os.path.join(self.save_dir, 'pde_error_heatmap.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Erro ao gerar heatmap: {e}")

    def plot_pde_residual_surface(self):
        """Superfície 3D do Resíduo."""
        try:
            x_phy, heston_params, S_grid, T_grid, asset_ids = self._generate_synthetic_grid()
            if x_phy is None: return
            
            # Recalcular preço com gradiente ligado para a PDE
            nu, theta, kappa, xi, rho = heston_params
            pinn_input = torch.cat([x_phy, nu, theta, kappa, xi, rho], dim=1)
            
            # Projeção e Rede
            feats = self.model.fourier_layer(pinn_input) if self.model.use_fourier else pinn_input
            nn_out = self.model.pricing_net(feats)
            
            # Hard Constraint (Reimplementação manual para derivar)
            T_norm = x_phy[:, 2:3]
            time_factor = 1.0 - torch.exp(-10.0 * T_norm)
            time_value = time_factor * torch.nn.functional.softplus(nn_out)
            
            # Payoff (S e K estão normalizados no grid)
            payoff = torch.relu(x_phy[:, 0:1] - x_phy[:, 1:2])
            price = payoff + time_value
            
            # Resíduo - CORRIGIDO: Adicionar return_residuals=True
            model_output = {'price': price, 'heston_params': heston_params}
            res = heston_residual(model_output, x_phy, self.data_stats, return_residuals=True)
            
            Z = self._safe_reshape_grid((res**2).detach().cpu().numpy(), S_grid.shape)
            
            # Verificação de NaNs
            if np.isnan(Z).any() or np.isinf(Z).any():
                logger.warning("NaNs ou Infs detectados na superfície de resíduo. Substituindo por 0 para plot.")
                Z = np.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)

            # Criação da Figura e Eixos 
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(S_grid, T_grid, Z, cmap='magma', edgecolor='none')
            ax.set_title('Erro da Física (Resíduo PDE Quadrático)')
            ax.set_xlabel('Spot')
            ax.set_ylabel('Tempo')
            fig.colorbar(surf, shrink=0.5)
            
            try:
                plt.savefig(os.path.join(self.save_dir, 'pde_residual_surface.png'))
            except OSError as e:
                logger.error(f"Falha de I/O ao salvar pde_residual: {e}")
            plt.close()
        except Exception as e:
            logger.error(f"Erro pde_residual: {e}")

    def plot_loss_history(self):
        """Plota histórico de loss de treino e validação."""
        try:
            if not os.path.exists(self.history_path):
                logger.warning("Arquivo de histórico não encontrado.")
                return
                
            df = pd.read_csv(self.history_path)
            
            # Usa índice se 'epoch' não existir
            x_axis = df['epoch'] if 'epoch' in df.columns else df.index + 1
            
            plt.figure(figsize=(10, 5))
            plt.plot(x_axis, df['train_loss'], label='Treino', linewidth=2)
            plt.plot(x_axis, df['val_loss'], label='Validação', linewidth=2, linestyle='--')
            
            # Verifica se pode usar log scale
            if (df['train_loss'] > 0).all() and (df['val_loss'] > 0).all():
                plt.yscale('log')
                plt.title('Evolução da Loss (Log Scale)')
            else:
                plt.title('Evolução da Loss (Linear Scale)')
                
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.savefig(os.path.join(self.save_dir, 'loss_history.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Erro ao plotar loss history: {e}")

    def plot_weights_history(self):
        """Plota evolução dos pesos adaptativos."""
        if not os.path.exists(self.history_path): return
        try:
            df = pd.read_csv(self.history_path)
            if 'weight_pde' in df.columns:
                plt.figure(figsize=(10, 5))
                
                # Adiciona pequeno epsilon para evitar log(0) e Warning
                w_pde = df['weight_pde'] + 1e-9 
                
                plt.plot(w_pde, label=r'Peso PDE ($\lambda_{phys}$)', color='orange')
                plt.title(r'Dinâmica de Curriculum Learning')
                plt.xlabel('Épocas')
                plt.ylabel('Peso')
                plt.yscale('log')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.save_dir, 'weights_history.png'))
                plt.close()
        except Exception as e: logger.error(f"Erro weights_history: {e}")

    def plot_all(self):
        """Gera todos os gráficos habilitados na configuração."""
        plt.close('all') # Limpeza preventiva
        logger.info(f"Iniciando geração de gráficos em: {self.save_dir}")
        
        jobs = [
            ('plot_loss', self.plot_loss_history),
            ('plot_weights_history', self.plot_weights_history),
            ('plot_premium_time', self.plot_premium_over_time), 
            ('plot_model_vs_bs', self.plot_model_vs_bs_comparison),
            ('plot_heston_params', self.plot_heston_params),
            ('plot_price_scatter', self.plot_prediction_scatter),
            ('plot_dist_overlay', self.plot_distribution_overlay),
            ('plot_residuals', self.plot_residuals),
            ('plot_error_by_moneyness', self.plot_error_by_moneyness),
            ('plot_error_distribution', self.plot_error_distribution),
            ('plot_error_heatmap', self.plot_error_heatmap),
            ('plot_price_surface', self.plot_price_surface),
            ('plot_delta_surface', self.plot_delta_surface),
            ('plot_pde_residual', self.plot_pde_residual_surface),
            ('plot_vol_smile', self.plot_vol_smile),
            ('plot_latent_vol', self.plot_latent_vol_evolution),
            ('plot_premium_by_moneyness_time', self.plot_premium_by_moneyness_time),
            ('price_surface', self.plot_price_surface),
            ('pde_residual', self.plot_pde_residual_surface),
            ('weights', self.plot_weights_history),

        ]
        
        for name, func in jobs:
            if self.config.get(name, True):
                try:
                    func()
                except Exception as e:
                    logger.error(f"Falha crítica ao gerar plot '{name}': {e}")
                    
        logger.info("Rotina de visualização finalizada.")