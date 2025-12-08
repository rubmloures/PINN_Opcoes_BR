# /src/visualization.py

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

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Cálculo analítico exato de Black-Scholes."""
    S = np.maximum(S, 1e-5)
    K = np.maximum(K, 1e-5)
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_implied_vol_approx(S, K, T, r, price):
    """Aproximação Brenner & Subrahmanyam para Vol Implícita."""
    T = np.maximum(T, 1e-5)
    S = np.maximum(S, 1e-5)
    price = np.maximum(price, 1e-5)
    return np.sqrt(2 * np.pi / T.flatten()) * (price.flatten() / S.flatten())

class Visualizer:
    def __init__(self, model: DeepHestonHybrid, history_path: str, val_loader, data_stats: dict, config: dict):
        self.model = model
        self.history_path = history_path
        self.val_loader = val_loader
        self.data_stats = data_stats
        self.config = config
        self.device = next(model.parameters()).device
        self.preds_cache = None
        
        # Diretório de salvamento
        self.save_dir = self.config.get('plot_save_dir', PATHS['plot_save_dir'])
        os.makedirs(self.save_dir, exist_ok=True)

    def _run_inference(self):
        """Roda inferência em todo o dataset de validação e faz cache dos resultados."""
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
        """Gera grid (S, T) sintético para plots de superfície e derivadas."""
        # Pega uma amostra real para gerar o estado inicial da LSTM
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
        
        # Converte para tensores [N*N, 1]
        S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=self.device).unsqueeze(1)
        T_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=self.device).unsqueeze(1)
        K_flat = torch.ones_like(S_flat) * 1.0   # Strike normalizado
        r_flat = torch.ones_like(S_flat) * 0.10  # Juros fixos 10%
        q_flat = torch.ones_like(S_flat) * 0.02  # Dividend yield fixo 2%
        
        # Input físico: [S, K, T, r, q]
        x_phy_synthetic = torch.cat([S_flat, K_flat, T_flat, r_flat, q_flat], dim=1).requires_grad_(True)        
        
        # Asset ID Fictício (Usa 0 como padrão para o plot genérico)
        asset_ids_synthetic = torch.zeros(x_phy_synthetic.shape[0], dtype=torch.long, device=self.device)

        # Parâmetros Heston (com gradiente habilitado para a EDP)
        # Detach para quebrar o grafo da LSTM (não precisamos derivar até a LSTM aqui)
        fixed_heston_params = (
            nu.detach().requires_grad_(True),
            theta.detach().requires_grad_(True),
            kappa.detach().requires_grad_(True),
            xi.detach().requires_grad_(True),
            rho.detach().requires_grad_(True)
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
        """Real vs Previsto (Scatter colorido por Tempo)."""
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
        """Plot 1: Evolução Temporal Simples (Global)."""
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
        """Plot 2 (NOVO): Evolução Temporal segmentada por Moneyness (ITM/ATM/OTM)."""
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
        """Comparação das distribuições de preços (Detecta colapso)."""
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
        """ Resíduos (Real - Previsto) vs Moneyness."""
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
        """MAE por categoria de Moneyness."""
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

    def plot_heston_params(self):
        """Distribuição dos parâmetros inferidos."""
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
        """Dinâmica da Volatilidade Estocástica."""
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
        """Superfície 3D do Delta (Greeks)."""
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
        """Comparação com Black-Scholes."""
        try:
            data = self._run_inference()
            idx = np.random.choice(len(data['price_pred']), min(5000, len(data['price_pred'])), replace=False)
            
            S = data['inputs_phy'][idx, 0] * (self.data_stats['S_max'] - self.data_stats['S_min']) + self.data_stats['S_min']
            K = data['inputs_phy'][idx, 1] * (self.data_stats['K_max'] - self.data_stats['K_min']) + self.data_stats['K_min']
            T = data['inputs_phy'][idx, 2] * self.data_stats['T_max']
            r = data['inputs_phy'][idx, 3]
            
            real = data['price_real'][idx]
            pred = data['price_pred'][idx]
            
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
        """Plot: Smile de Volatilidade (Corrigido e Expandido)."""
        try:
            data = self._run_inference()
            # Amostra aleatória para não pesar
            idx = np.random.choice(len(data['price_pred']), min(8000, len(data['price_pred'])), replace=False)
            
            S = data['inputs_phy'][idx, 0] * (self.data_stats['S_max'] - self.data_stats['S_min']) + self.data_stats['S_min']
            K = data['inputs_phy'][idx, 1] * (self.data_stats['K_max'] - self.data_stats['K_min']) + self.data_stats['K_min']
            T = data['inputs_phy'][idx, 2] * self.data_stats['T_max']
            r = data['inputs_phy'][idx, 3]
            prices = data['price_pred'][idx]
            
            # Filtro mais permissivo: Maturidade entre 2 semanas e 6 meses
            mask = (T > 0.04) & (T < 0.5)
            if mask.sum() < 50: return

            iv = bs_implied_vol_approx(S[mask], K[mask], T[mask], r[mask], prices[mask])
            mon = K[mask] / S[mask]
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
                # Curto prazo
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