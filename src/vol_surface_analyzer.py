# /src/vol_surface_analyzer.py
"""
Analisador de Superfície de Volatilidade Implícita.
Extrai IV do modelo usando inversão de Black-Scholes.
Analisa smile, skew, e estrutura a termo.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq
import seaborn as sns

from src.logger import get_logger

logger = get_logger('VolSurfaceAnalyzer')

class VolSurfaceAnalyzer:
    """Analisa e visualiza superfícies de volatilidade implícita."""
    
    def __init__(self, model, data_stats, config, paths):
        """
        Args:
            model: Modelo treinado
            data_stats: Estatísticas de normalização
            config: Configurações
            paths: Caminhos de diretórios
        """
        self.model = model
        self.data_stats = data_stats
        self.config = config
        self.paths = paths
        self.device = next(model.parameters()).device
        
    def black_scholes_call(self, S: float, K: float, T: float, r: float, 
                          sigma: float, q: float = 0.0) -> float:
        """
        Fórmula fechada de Black-Scholes para Call Option.
        
        Args:
            S: Spot price
            K: Strike
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatilidade (IV)
            q: Dividend yield
            
        Returns:
            Preço da opção Call
        """
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        from scipy.stats import norm
        price = (S * np.exp(-q * T) * norm.cdf(d1) - 
                K * np.exp(-r * T) * norm.cdf(d2))
        
        return price
    
    def implied_volatility(self, S: float, K: float, T: float, r: float, 
                          market_price: float, q: float = 0.0,
                          sigma_min: float = 0.001, sigma_max: float = 2.0) -> float:
        """
        Calcula IV usando Brent method (inversão de Black-Scholes).
        
        Args:
            S, K, T, r, q: Parâmetros da opção
            market_price: Preço de mercado (ou previsão do modelo)
            sigma_min, sigma_max: Limites de busca
            
        Returns:
            Volatilidade implícita ou NaN se não convergir
        """
        try:
            def objective(sigma):
                bs_price = self.black_scholes_call(S, K, T, r, sigma, q)
                return bs_price - market_price
            
            # Verifica se o preço está dentro do intervalo possível
            price_min = self.black_scholes_call(S, K, T, r, sigma_min, q)
            price_max = self.black_scholes_call(S, K, T, r, sigma_max, q)
            
            if market_price < price_min - 1e-6 or market_price > price_max + 1e-6:
                return np.nan
            
            # Busca a IV
            iv = brentq(objective, sigma_min, sigma_max, xtol=1e-6)
            return iv
        except:
            return np.nan
    
    def generate_vol_surface(self, S: float, T_list: List[float], 
                            K_list: List[float], r: float = 0.105, 
                            asset_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gera superfície completa de volatilidade implícita.
        
        Args:
            S: Spot price inicial
            T_list: Lista de tempos para maturação
            K_list: Lista de strikes
            r: Taxa de juros
            asset_id: ID do ativo (para embedding)
            
        Returns:
            (T_mesh, K_mesh, IV_mesh)
        """
        self.model.eval()
        T_mesh, K_mesh = np.meshgrid(T_list, K_list)
        IV_mesh = np.zeros_like(T_mesh, dtype=float)
        
        logger.info(f"\nGerando superfície de volatilidade:")
        logger.info(f"  S = {S:.2f}")
        logger.info(f"  T ∈ [{T_list[0]:.3f}, {T_list[-1]:.3f}] ({len(T_list)} pontos)")
        logger.info(f"  K/S ∈ [{K_list[0]/S:.3f}, {K_list[-1]/S:.3f}] ({len(K_list)} strikes)")
        
        total_points = len(T_list) * len(K_list)
        processed = 0
        
        with torch.no_grad():
            for i, T in enumerate(T_list):
                for j, K in enumerate(K_list):
                    # Normaliza inputs
                    S_norm = (S - self.data_stats['S_mean']) / self.data_stats['S_std']
                    K_norm = (K - self.data_stats['K_mean']) / self.data_stats['K_std']
                    T_norm = (T - self.data_stats['T_mean']) / self.data_stats['T_std']
                    r_norm = (r - self.data_stats['r_mean']) / self.data_stats['r_std']
                    q_norm = (0.0 - self.data_stats['q_mean']) / self.data_stats['q_std']
                    
                    x_phy = torch.tensor(
                        [[S_norm, K_norm, T_norm, r_norm, q_norm]], 
                        dtype=torch.float32, device=self.device
                    )
                    
                    # LSTM input (dummy - não vai afetar muito se não tiver histórico real)
                    x_seq = torch.zeros((1, 30, self.config.get('lstm_input_size', 6)), 
                                      dtype=torch.float32, device=self.device)
                    
                    asset_id_tensor = torch.tensor([asset_id], dtype=torch.long, device=self.device)
                    
                    # Forward
                    output = self.model(x_seq, x_phy, asset_id_tensor)
                    price = output['price'].cpu().item()
                    
                    # Calcula IV
                    iv = self.implied_volatility(S, K, T, r, max(price, 0.001), q=0.0)
                    IV_mesh[j, i] = iv if not np.isnan(iv) else 0.0
                    
                    processed += 1
                    if processed % max(1, total_points // 10) == 0:
                        logger.info(f"  Progresso: {processed}/{total_points} ({100*processed/total_points:.0f}%)")
        
        logger.info(f"  ✅ Superfície gerada: {total_points} pontos")
        return T_mesh, K_mesh, IV_mesh
    
    def analyze_vol_smile(self, T_mesh: np.ndarray, K_mesh: np.ndarray, 
                         IV_mesh: np.ndarray, S: float) -> Dict:
        """
        Analisa características do smile (convexidade, assimetria).
        
        Args:
            T_mesh, K_mesh, IV_mesh: Superfície gerada
            S: Spot price para calcular moneyness
            
        Returns:
            Dict com métricas de smile/skew
        """
        T_unique = T_mesh[0, :]
        K_unique = K_mesh[:, 0]
        moneyness = K_unique / S
        
        metrics = {
            'smile_by_maturity': {},
            'skew_by_maturity': {},
            'average_smile': None,
        }
        
        # Analisa cada maturidade
        for k in range(IV_mesh.shape[1]):
            T = T_unique[k]
            IV_slice = IV_mesh[:, k]
            
            # Encontra mínimo (ATM) e máximos (wings)
            atm_idx = np.argmin(np.abs(moneyness - 1.0))
            atm_iv = IV_slice[atm_idx]
            
            # Calcula smile: diferença entre wings e ATM
            wing_mask = (moneyness < 0.95) | (moneyness > 1.05)
            if wing_mask.sum() > 0:
                wing_iv = np.mean(IV_slice[wing_mask])
                smile = wing_iv - atm_iv
            else:
                smile = 0
            
            # Calcula skew: diferença entre put OTM vs call OTM
            put_otm_mask = (moneyness >= 0.85) & (moneyness < 0.95)
            call_otm_mask = (moneyness > 1.05) & (moneyness <= 1.15)
            
            put_iv = np.mean(IV_slice[put_otm_mask]) if put_otm_mask.sum() > 0 else np.nan
            call_iv = np.mean(IV_slice[call_otm_mask]) if call_otm_mask.sum() > 0 else np.nan
            
            skew = call_iv - put_iv if not (np.isnan(put_iv) or np.isnan(call_iv)) else np.nan
            
            metrics['smile_by_maturity'][f'T={T:.3f}'] = {
                'ATM_IV': atm_iv,
                'Smile': smile,
                'Skew': skew,
            }
        
        # Média geral
        all_smiles = [v.get('Smile', 0) for v in metrics['smile_by_maturity'].values()]
        metrics['average_smile'] = np.mean([s for s in all_smiles if not np.isnan(s)])
        
        return metrics
    
    def plot_vol_surface_3d(self, T_mesh: np.ndarray, K_mesh: np.ndarray, 
                           IV_mesh: np.ndarray, asset_name: str = 'Asset'):
        """Visualiza superfície 3D de volatilidade."""
        fig = plt.figure(figsize=(14, 5))
        
        # Plot 3D
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(T_mesh, K_mesh, IV_mesh, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Time to Maturity (years)')
        ax1.set_ylabel('Strike')
        ax1.set_zlabel('Implied Volatility')
        ax1.set_title(f'Vol Surface: {asset_name}')
        fig.colorbar(surf, ax=ax1)
        
        # Plot 2D (contour)
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(T_mesh, K_mesh, IV_mesh, levels=20, cmap='viridis')
        ax2.contour(T_mesh, K_mesh, IV_mesh, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax2.set_xlabel('Time to Maturity (years)')
        ax2.set_ylabel('Strike')
        ax2.set_title(f'Vol Surface (Contour): {asset_name}')
        fig.colorbar(contour, ax=ax2)
        
        fig.tight_layout()
        
        # Salva
        save_dir = os.path.join(self.paths.get('results_dir', 'resultados'), 
                               'vol_surface_analysis')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'vol_surface_3d_{asset_name}.png'), 
                   dpi=150, bbox_inches='tight')
        logger.info(f"Figura salva: {save_dir}/vol_surface_3d_{asset_name}.png")
        plt.close(fig)
    
    def plot_vol_smile(self, T_mesh: np.ndarray, K_mesh: np.ndarray, 
                      IV_mesh: np.ndarray, S: float, asset_name: str = 'Asset'):
        """Visualiza vol smile para diferentes maturidades."""
        T_unique = T_mesh[0, :]
        K_unique = K_mesh[:, 0]
        moneyness = K_unique / S
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Vol Smile Analysis: {asset_name}', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        selected_indices = np.linspace(0, IV_mesh.shape[1]-1, 4, dtype=int)
        
        for idx, T_idx in enumerate(selected_indices):
            T = T_unique[T_idx]
            IV_slice = IV_mesh[:, T_idx]
            
            ax = axes[idx]
            ax.plot(moneyness, IV_slice, 'b-', linewidth=2, marker='o', markersize=4)
            ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='ATM')
            ax.fill_between(moneyness, IV_slice, alpha=0.2)
            ax.set_xlabel('Moneyness (K/S)')
            ax.set_ylabel('Implied Volatility')
            ax.set_title(f'T = {T:.3f} years')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim([moneyness.min(), moneyness.max()])
        
        fig.tight_layout()
        
        # Salva
        save_dir = os.path.join(self.paths.get('results_dir', 'resultados'), 
                               'vol_surface_analysis')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'vol_smile_{asset_name}.png'), 
                   dpi=150, bbox_inches='tight')
        logger.info(f"Figura salva: {save_dir}/vol_smile_{asset_name}.png")
        plt.close(fig)
    
    def generate_report(self, S: float, asset_name: str = 'Asset', 
                       asset_id: int = 0) -> Dict:
        """
        Gera relatório completo de análise de volatilidade.
        
        Args:
            S: Spot price
            asset_name: Nome do ativo
            asset_id: ID para embedding
            
        Returns:
            Dict com superfície e análises
        """
        logger.info("\n" + "█"*80)
        logger.info("█" + " "*20 + f"ANÁLISE DE SUPERFÍCIE DE VOLATILIDADE" + " "*20 + "█")
        logger.info("█"*80)
        logger.info(f"Ativo: {asset_name} | Spot: {S:.2f}")
        
        # Cria grid
        T_list = np.linspace(0.05, 2.5, 25)  # 5 dias a 2.5 anos
        K_list = np.linspace(S * 0.75, S * 1.25, 25)  # Deep OTM a Deep ITM
        
        # Gera superfície
        T_mesh, K_mesh, IV_mesh = self.generate_vol_surface(
            S, T_list, K_list, asset_id=asset_id
        )
        
        # Análise de smile
        metrics = self.analyze_vol_smile(T_mesh, K_mesh, IV_mesh, S)
        
        logger.info(f"\n📊 Análise de Smile/Skew:")
        logger.info(f"  Smile Médio: {metrics['average_smile']:.4f}")
        logger.info(f"  (Positivo = Convexidade tipo equity, Negativo = Reverso)")
        
        # Visualizações
        logger.info(f"\n🎨 Gerando visualizações...")
        self.plot_vol_surface_3d(T_mesh, K_mesh, IV_mesh, asset_name)
        self.plot_vol_smile(T_mesh, K_mesh, IV_mesh, S, asset_name)
        
        # Salva dados
        save_dir = os.path.join(self.paths.get('results_dir', 'resultados'), 
                               'vol_surface_analysis')
        os.makedirs(save_dir, exist_ok=True)
        
        df_surface = pd.DataFrame({
            'T': T_mesh.flatten(),
            'K': K_mesh.flatten(),
            'IV': IV_mesh.flatten(),
            'Moneyness': (K_mesh.flatten() / S),
        })
        df_surface.to_csv(os.path.join(save_dir, f'vol_surface_{asset_name}.csv'), index=False)
        
        logger.info(f"✅ Dados salvos em: {save_dir}/vol_surface_{asset_name}.csv")
        
        return {
            'T_mesh': T_mesh,
            'K_mesh': K_mesh,
            'IV_mesh': IV_mesh,
            'metrics': metrics,
        }


if __name__ == '__main__':
    # Exemplo de uso
    pass
