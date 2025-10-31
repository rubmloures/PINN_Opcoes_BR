# /src/visualization.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from torch.utils.data import DataLoader
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from src.model import PINN_BlackScholes
from src.physics import black_scholes_residual
from src.config import VIZ_CONFIG, PATHS, DATA_CONFIG

def black_scholes_call_numpy(S, K, T, r, sigma):
    """Função analítica de Black-Scholes em NumPy para plotagem de superfície."""
    # Adiciona um pequeno epsilon para evitar divisão por zero
    T = np.where(T <= 1e-8, 1e-8, T)
    sigma = np.where(sigma <= 1e-8, 1e-8, sigma)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

class Visualizer:
    def __init__(self, model: PINN_BlackScholes, history_path: str, val_loader: DataLoader, data_stats: dict, config: dict):
        self.model = model
        self.val_loader = val_loader  
        self.data_stats = data_stats
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_weights_path = os.path.join(PATHS['model_save_dir'], 'best_model_weights.pth')
        
        if os.path.exists(best_weights_path):
            self.model.load_state_dict(torch.load(best_weights_path, map_location=self.device))
            print("Pesos do melhor modelo carregados para visualização.")
        else:
            print(f"Aviso: Arquivo de pesos do melhor modelo não encontrado em {best_weights_path}.")
            
        if os.path.exists(history_path):
             self.history = pd.read_csv(history_path).to_dict('list')
        else:
            print(f"Aviso: Arquivo de histórico '{history_path}' não encontrado.")
            self.history = None

        self.model.to(self.device)
        self.model.eval()
        os.makedirs(PATHS['plot_save_dir'], exist_ok=True)

    # Plot 1 
    def plot_loss_history(self):
        if not self.history:
            print("Não foi possível gerar o gráfico de perdas: histórico não disponível.")
            return
            
        print("Gerando gráfico do histórico de perdas...")
        fig, ax1 = plt.subplots(figsize=(12, 7))

        ax1.plot(self.history['train_loss'], label='Perda de Treinamento', color='blue')
        ax1.plot(self.history['val_loss'], label='Perda de Validação', color='orange')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss (MSE)', color='blue')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, which="both", ls="--", axis='y')
        
        ax2 = ax1.twinx()
        ax2.plot(self.history['lr'], label='Taxa de Aprendizado (LR)', color='green', linestyle='--')
        ax2.set_ylabel('Taxa de Aprendizado', color='green')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='green')

        fig.suptitle('Histórico de Perdas e Taxa de Aprendizado')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        
        save_path = os.path.join(PATHS['plot_save_dir'], 'loss_history.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico de perdas salvo em: {save_path}")

    # Plot 2 
    def _prepare_surface_data(self, resolution=50): # Resolução menor para testes mais rápidos
        moneyness_vals = torch.linspace(0.8, 1.2, resolution, device=self.device)
        T_norm_vals = torch.linspace(0.01, 1, resolution, device=self.device)
        
        M_grid, T_grid = torch.meshgrid(moneyness_vals, T_norm_vals, indexing='ij')
        
        K_fixed = (self.data_stats['K_max'] + self.data_stats['K_min']) / 2
        S_vals = M_grid * K_fixed

        S_norm_grid = (S_vals - self.data_stats['S_min']) / (self.data_stats['S_max'] - self.data_stats['S_min'])
        K_norm_fixed = (K_fixed - self.data_stats['K_min']) / (self.data_stats['K_max'] - self.data_stats['K_min'])
        K_norm_grid = torch.full_like(S_norm_grid, K_norm_fixed)

        r_grid = torch.full_like(S_norm_grid, 0.13)
        premium_grid = torch.full_like(S_norm_grid, 1.0)

        model_input = torch.stack([
            S_norm_grid.flatten(), K_norm_grid.flatten(), T_grid.flatten(), 
            r_grid.flatten(), premium_grid.flatten()
        ], dim=1)
        
        return M_grid, T_grid, model_input

    # Plot 
    def plot_implied_volatility_surface(self):
        print("Gerando superfície de volatilidade implícita...")
        M_grid, T_grid, model_input = self._prepare_surface_data()
        
        with torch.no_grad():
            sigma_pred = self.model(model_input)['sigma'].reshape_as(M_grid) * 100

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        M_cpu = M_grid.cpu().numpy()
        T_cpu = T_grid.cpu().numpy() * self.data_stats['T_max'] * 252
        sigma_cpu = sigma_pred.cpu().numpy()

        surf = ax.plot_surface(M_cpu, T_cpu, sigma_cpu, cmap='viridis', edgecolor='none')
        
        ax.set_title('Superfície de Volatilidade Implícita Aprendida (PINN)')
        ax.set_xlabel('Moneyness (S/K)')
        ax.set_ylabel('Dias para o Vencimento')
        ax.set_zlabel('Volatilidade Implícita (%)')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        save_path = os.path.join(PATHS['plot_save_dir'], 'implied_volatility_surface.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico da superfície de volatilidade salvo em: {save_path}")
        
    # Plot 3    
    def plot_pde_residual_surface(self):
        print("Gerando superfície de resíduo da EDP...")
        M_grid, T_grid, model_input = self._prepare_surface_data()
        model_input.requires_grad = True

        output_phy = self.model(model_input)
        residual = black_scholes_residual(output_phy, model_input, self.data_stats)
        
        residual_abs = torch.abs(residual).reshape_as(M_grid)

        fig, ax = plt.subplots(figsize=(12, 8))
        
        M_cpu = M_grid.cpu().numpy()
        T_cpu = T_grid.cpu().numpy() * self.data_stats['T_max'] * 252
        residual_cpu = residual_abs.cpu().detach().numpy()

        c = ax.pcolormesh(M_cpu, T_cpu, residual_cpu, cmap='hot', shading='gouraud', vmax=np.percentile(residual_cpu, 99))
        ax.set_title('Mapa de Calor do Resíduo Absoluto da EDP de Black-Scholes')
        ax.set_xlabel('Moneyness (S/K)')
        ax.set_ylabel('Dias para o Vencimento')
        fig.colorbar(c, ax=ax, label='|Resíduo da EDP|')
        
        save_path = os.path.join(PATHS['plot_save_dir'], 'pde_residual_surface.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico do resíduo da EDP salvo em: {save_path}")
    
    # Plot 4
    def plot_prediction_vs_actual(self):
        print("Gerando gráfico de Preço Previsto vs. Preço Real...")
        self.model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for inputs, premiums_real in self.val_loader: # Usa self.val_loader
                inputs = inputs.to(self.device)
                price_pred = self.model(inputs)['price']
                actuals.extend(premiums_real.cpu().numpy())
                predictions.extend(price_pred.cpu().numpy())
        
        plt.figure(figsize=(10, 10))
        plt.scatter(actuals, predictions, alpha=0.3, label='Previsões do Modelo')
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Linha de Perfeição (y=x)')
        plt.xlabel('Preço de Mercado Real (Prêmio)')
        plt.ylabel('Preço Previsto pela PINN')
        plt.title('Comparação entre Preço Real e Previsto (Conjunto de Validação)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        save_path = os.path.join(PATHS['plot_save_dir'], 'prediction_vs_actual.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico de comparação de preços salvo em: {save_path}")
    
    # Plot 5
    def plot_error_by_moneyness(self):
        print("Gerando gráfico de Erro por Moneyness...")
        self.model.eval()
        moneyness_vals = []
        errors = []
        with torch.no_grad():
            for inputs, premiums_real in self.val_loader: # Usa self.val_loader
                S_norm, K_norm = inputs[:, 0], inputs[:, 1]
                S = S_norm * (self.data_stats['S_max'] - self.data_stats['S_min']) + self.data_stats['S_min']
                K = K_norm * (self.data_stats['K_max'] - self.data_stats['K_min']) + self.data_stats['K_min']
                moneyness = (S / K).cpu().numpy()
                
                inputs = inputs.to(self.device)
                price_pred = self.model(inputs)['price'].cpu().numpy()
                
                valid_indices = premiums_real.numpy().flatten() > 1e-3
                error_pct = (price_pred.flatten()[valid_indices] - premiums_real.numpy().flatten()[valid_indices]) / premiums_real.numpy().flatten()[valid_indices]
                
                moneyness_vals.extend(moneyness[valid_indices])
                errors.extend(error_pct)
                
        plt.figure(figsize=(12, 7))
        plt.scatter(moneyness_vals, errors, alpha=0.1)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Moneyness (S/K)')
        plt.ylabel('Erro Percentual de Previsão ((Prev - Real) / Real)')
        plt.title('Distribuição do Erro de Previsão por Moneyness')
        plt.grid(True)
        save_path = os.path.join(PATHS['plot_save_dir'], 'error_by_moneyness.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico de erro por moneyness salvo em: {save_path}")

    # Plot 6
    def plot_greeks_surface(self):
        print("Gerando superfície do Delta (∂V/∂S)...")
        M_grid, T_grid, model_input = self._prepare_surface_data()
        model_input.requires_grad = True

        output = self.model(model_input)
        V = output['price']

        V_grads = torch.autograd.grad(V, model_input, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        V_S_norm = V_grads[:, 0]
        
        delta = V_S_norm / (self.data_stats['S_max'] - self.data_stats['S_min'])
        delta_surface = delta.reshape_as(M_grid)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        M_cpu = M_grid.cpu().detach().numpy()
        T_cpu = T_grid.cpu().detach().numpy() * self.data_stats['T_max'] * 252
        delta_cpu = delta_surface.cpu().detach().numpy()
        
        surf = ax.plot_surface(M_cpu, T_cpu, delta_cpu, cmap='cividis', edgecolor='none')
        ax.set_title('Superfície do Delta Aprendido pela PINN')
        ax.set_xlabel('Moneyness (S/K)')
        ax.set_ylabel('Dias para o Vencimento')
        ax.set_zlabel('Delta (∂V/∂S)')
        ax.set_zlim(0, 1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        save_path = os.path.join(PATHS['plot_save_dir'], 'delta_surface.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico da superfície do Delta salvo em: {save_path}")

    # Plot 7
    def plot_price_surfaces_comparison(self):
        print("Gerando comparação de superfícies de preço (PINN vs. Analítico)...")
        resolution=40
        s_vals = np.linspace(self.data_stats['S_min'], self.data_stats['S_max'], resolution)
        t_vals = np.linspace(0.01, self.data_stats['T_max'], resolution)
        S_grid, T_grid = np.meshgrid(s_vals, t_vals)
        K_fixed = (self.data_stats['K_max'] + self.data_stats['K_min']) / 2
        r_fixed = 0.13
        C_pinn = np.zeros_like(S_grid)
        sigma_pinn_grid = np.zeros_like(S_grid)
        with torch.no_grad():
            for i in range(resolution):
                for j in range(resolution):
                    S, T = S_grid[i, j], T_grid[i, j]
                    S_n = (S - self.data_stats['S_min']) / (self.data_stats['S_max'] - self.data_stats['S_min'])
                    T_n = T / self.data_stats['T_max']
                    K_n = (K_fixed - self.data_stats['K_min']) / (self.data_stats['K_max'] - self.data_stats['K_min'])
                    inp = torch.tensor([[S_n, K_n, T_n, r_fixed, 1.0]], dtype=torch.float32).to(self.device)
                    output = self.model(inp)
                    C_pinn[i, j] = output['price'].item()
                    sigma_pinn_grid[i, j] = output['sigma'].item()
        C_bs = black_scholes_call_numpy(S_grid, K_fixed, T_grid, r_fixed, np.mean(sigma_pinn_grid))
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(S_grid, T_grid * 252, C_pinn, cmap='viridis')
        ax1.set_title("Superfície de Preço da PINN")
        ax1.set_xlabel('Preço do Ativo (S)'); ax1.set_ylabel('Dias para Vencimento'); ax1.set_zlabel('Preço da Opção (V)')
        ax2.plot_surface(S_grid, T_grid * 252, C_bs, cmap='plasma')
        ax2.set_title("Superfície Analítica de Black-Scholes (Vol Média da PINN)")
        ax2.set_xlabel('Preço do Ativo (S)'); ax2.set_ylabel('Dias para Vencimento'); ax2.set_zlabel('Preço da Opção (V)')
        plt.tight_layout()
        save_path = os.path.join(PATHS['plot_save_dir'], 'price_surfaces_comparison.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico de comparação de superfícies salvo em: {save_path}")

    # Plot 8
    def plot_volatility_comparison(self):
        print("Gerando gráfico de Volatilidade Prevista vs. Volatilidade Real (Histórica)...")
        self.model.eval()
        vol_real_list = []
        vol_pred_list = []
        with torch.no_grad():
            for inputs, _ in self.val_loader:
                # A volatilidade real (histórica) não é usada no input, mas a temos no dataset
                # Vamos reconstruir o input sem a vol para o modelo, mas guardar a vol para comparação
                # A 5ª coluna (índice 4) é o prêmio, a vol não está aqui. Precisamos carregar os dados completos.
                pass # Esta lógica precisa ser ajustada para ter acesso à volatilidade original

        # SIMPLIFICAÇÃO: Para este exemplo, vamos plotar a vol prevista em relação ao moneyness
        M_grid, _, model_input = self._prepare_surface_data(resolution=100)
        with torch.no_grad():
            sigma_pred = self.model(model_input)['sigma'].reshape_as(M_grid) * 100
        
        plt.figure(figsize=(10, 7))
        # Plotamos um "corte" da superfície de volatilidade para um vencimento médio
        T_median_idx = sigma_pred.shape[1] // 2
        plt.plot(M_grid[:, T_median_idx].cpu().numpy(), sigma_pred[:, T_median_idx].cpu().numpy(), label=f'Corte da Vol. Implícita (PINN)')
        
        plt.xlabel('Moneyness (S/K)')
        plt.ylabel('Volatilidade Implícita (%)')
        plt.title('Curva de Volatilidade Implícita Aprendida (Vencimento Fixo)')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(PATHS['plot_save_dir'], 'volatility_smile.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico de curva de volatilidade salvo em: {save_path}")

    # Plot 9
    def plot_weights_history(self):
        """(NOVO) Plota a evolução dos pesos da perda ao longo do tempo."""
        if not self.history or 'weight_data' not in self.history or 'weight_pde' not in self.history:
            print("Não foi possível gerar o gráfico de pesos: histórico não disponível ou incompleto.")
            return
            
        print("Gerando gráfico da evolução dos pesos da perda...")
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Eixo esquerdo para o peso dos dados
        ax1.plot(self.history['weight_data'], label='Peso da Perda de Dados (λ_data)', color='deepskyblue')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Peso da Perda de Dados', color='deepskyblue')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='deepskyblue')
        ax1.grid(True, which="both", ls="--", axis='y')
        
        # Eixo direito para o peso da física (PDE)
        ax2 = ax1.twinx()
        ax2.plot(self.history['weight_pde'], label='Peso da Perda da Física (λ_pde)', color='crimson', linestyle='--')
        ax2.set_ylabel('Peso da Perda da Física', color='crimson')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='crimson')

        fig.suptitle('Evolução dos Pesos da Perda (Ponderação Adaptativa)')
        # Coleta os handles e labels de ambos os eixos para uma legenda unificada
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper center')
        
        save_path = os.path.join(PATHS['plot_save_dir'], 'weights_history.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico de pesos da perda salvo em: {save_path}")

    # Plot 10
    #...

    # Chamadas das plotagens
    def plot_all(self):
        if self.config.get('plot_loss', False):
            self.plot_loss_history()
        if self.config.get('plot_vol_surface', False):
            self.plot_implied_volatility_surface()
        if self.config.get('plot_pde_residual', False):
            self.plot_pde_residual_surface()
        if self.config.get('plot_price_comparation', False):
            self.plot_prediction_vs_actual()
        if self.config.get('plot_moneyness_comparation', False):
            self.plot_error_by_moneyness()
        if self.config.get('plot_greeks_comparation', False):
            self.plot_greeks_surface()
        if self.config.get('plot_price_surfaces', False): 
            self.plot_price_surfaces_comparison()
        if self.config.get('plot_vol_smile', False): 
            self.plot_volatility_comparison()
        if self.config.get('plot_weights_history', False): 
            self.plot_weights_history()

