# /src/visualization.py
"""
Módulo de Visualização para Physics-Informed Neural Networks (PINN)
Reformulado com foco em validação de consistência física para precificação de opções

Integrado ao pipeline Heston-LSTM Híbrido
Baseado em métricas de validação física:
- Resíduo da Equação Diferencial (PDE Residual)
- Consistência das Gregas (Delta, Gamma)
- Condições de Contorno e Terminais
- Quantificação de Incerteza
- Restrições de Não-Arbitragem
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from scipy.stats import norm
import warnings

# Plotly para gráficos interativos
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Importações do pipeline
from src.config import VIZ_CONFIG, PATHS, DATA_CONFIG, MODEL_CONFIG
from src.physics import heston_residual, PhysicsUtils, payoff_boundary_condition
from src.logger import get_logger

# Configurar logger
logger = get_logger('PINN_Visualization')

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# SEÇÃO 1: Funções Utilitárias Financeiras
# ==============================================================================

def black_scholes_price(S, K, T, r, sigma, q=0.0, option_type='call'):
    """
    Calcula o preço teórico de uma opção Europeia via Black-Scholes-Merton.
    Usado como baseline para comparação.
    
    Args:
        S: Preço Spot do ativo
        K: Preço de Exercício (Strike)
        T: Tempo até o vencimento (em anos)
        r: Taxa livre de risco anualizada
        sigma: Volatilidade implícita
        q: Dividend yield (default: 0.0)
        option_type: 'call' ou 'put'
    
    Returns:
        Preço teórico da opção
    """
    # Proteção numérica
    S = np.maximum(S, 1e-5)
    K = np.maximum(K, 1e-5)
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)
    
    # Cálculo de d1 e d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price

def bs_implied_vol_approx(S, K, T, r, price, q=0.0):
    """
    Aproximação de Brenner & Subrahmanyam para Volatilidade Implícita.
    
    Args:
        S, K, T, r, price, q: Parâmetros da opção
    
    Returns:
        Volatilidade implícita aproximada
    """
    T = np.maximum(T, 1e-5)
    S = np.maximum(S, 1e-5)
    price = np.maximum(price, 1e-5)
    
    return np.sqrt(2 * np.pi / T.flatten()) * (price.flatten() / S.flatten())

# ==============================================================================
# SEÇÃO 2: Classe Visualizer (Motor Principal)
# ==============================================================================

class Visualizer:
    """
    Gerenciador central de visualizações do Pipeline PINN.
    Foco em validação física e métricas de convergência.
    """
    
    def __init__(self, model, history_path: str, val_loader, data_stats: dict, config: dict):
        """
        Inicializa o visualizador com contexto do pipeline.
        
        Args:
            model: Modelo DeepHestonHybrid treinado
            history_path: Caminho para training_history.csv
            val_loader: DataLoader de validação
            data_stats: Estatísticas de normalização
            config: Configuração de visualização (VIZ_CONFIG)
        """
        self.model = model
        self.val_loader = val_loader
        self.data_stats = data_stats
        self.config = config
        self.device = next(model.parameters()).device
        
        # Diretório de salvamento
        self.save_dir = PATHS.get('plot_save_dir', 'resultados/plots')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Carregar histórico de treinamento
        self.history = None
        if os.path.exists(history_path):
            try:
                self.history = pd.read_csv(history_path)
                logger.info(f"Histórico carregado: {len(self.history)} épocas")
            except Exception as e:
                logger.warning(f"Erro ao carregar histórico: {e}")
        else:
            logger.warning(f"Histórico não encontrado: {history_path}")
        
        # Modo de avaliação
        self.model.eval()
        
        logger.info(f"Visualizer inicializado. Plots serão salvos em: {self.save_dir}")
    
    def _run_inference(self):
        """
        Roda inferência em todo o dataset de validação e faz cache dos resultados.
        Essencial para performance, evitando múltiplos loops no DataLoader.
        
        Returns:
            dict com 'price_pred', 'price_real', 'heston_params', 'inputs_phy', 'times'
        """
        if hasattr(self, 'preds_cache') and self.preds_cache is not None:
            return self.preds_cache

        self.model.eval()
        results = {'pred': [], 'real': [], 'params': [], 'phy': [], 'times': []}

        logger.info("Executando inferência completa no dataset de validação...")
        with torch.no_grad():
            for batch in self.val_loader:
                # Desempacotar batch - pode ter 3, 4 ou 6 elementos
                if len(batch) >= 6:
                    x_seq = batch[0].to(self.device)
                    x_phy = batch[1].to(self.device)
                    y_real = batch[2].to(self.device)
                    times = batch[3]  # não mover para device, é numpy
                    asset_ids = batch[5].to(self.device)
                elif len(batch) == 4:
                    x_seq = batch[0].to(self.device)
                    x_phy = batch[1].to(self.device)
                    y_real = batch[2].to(self.device)
                    asset_ids = batch[3].to(self.device)
                    times = None
                else:
                    x_seq = batch[0].to(self.device)
                    x_phy = batch[1].to(self.device)
                    y_real = batch[2].to(self.device)
                    asset_ids = torch.zeros(len(y_real), dtype=torch.long).to(self.device)
                    times = None

                outputs = self.model(x_seq, x_phy, asset_ids)
                
                results['pred'].append(outputs['price'].cpu().numpy())
                results['real'].append(y_real.cpu().numpy())
                
                # Extrair parâmetros Heston - agora está em outputs
                if 'heston_params' in outputs:
                    params_tuple = outputs['heston_params']
                    # params_tuple é (nu, theta, kappa, xi, rho) - 5 tensores
                    params_concat = torch.stack(params_tuple, dim=1)  # [batch, 5]
                    results['params'].append(params_concat.cpu().numpy())
                
                results['phy'].append(x_phy.cpu().numpy())
                
                if times is not None:
                    results['times'].append(times.numpy() if torch.is_tensor(times) else times)

        # Concatenação segura
        self.preds_cache = {
            'price_pred': np.concatenate(results['pred']),
            'price_real': np.concatenate(results['real']),
            'heston_params': np.concatenate(results['params']) if results['params'] else None,
            'inputs_phy': np.concatenate(results['phy']),
            'times': np.concatenate(results['times']) if results['times'] else None
        }
        return self.preds_cache
    
    def get_predictions(self):
        """
        Wrapper para compatibilidade: retorna 6 arrays separados.
        Usa _run_inference internamente.
        
        Returns:
            (y_real, y_pred, S, K, T, params) - 6 arrays numpy
        """
        data = self._run_inference()
        
        # Desnormalizar
        y_real = data['price_real'].flatten()
        y_pred = data['price_pred'].flatten()
        
        S = data['inputs_phy'][:, 0].flatten()
        K = data['inputs_phy'][:, 1].flatten()
        T = data['inputs_phy'][:, 2].flatten()
        
        # Desnormalizar S, K, T
        S = S * self.data_stats['S_std'] + self.data_stats['S_mean']
        K = K * self.data_stats['K_std'] + self.data_stats['K_mean']
        T = T * self.data_stats['T_std'] + self.data_stats['T_mean']
        
        params = data['heston_params'] if data['heston_params'] is not None else np.zeros((len(y_real), 5))
        
        return y_real, y_pred, S, K, T, params
    
    def run_inference(self):
        """
        Adaptador: Executa get_predictions e retorna um DataFrame pandas completo.
        Resolve o erro: 'Visualizer' object has no attribute 'run_inference'
        """
        # Chama a função existente que retorna 6 valores
        y_real, y_pred, S, K, T, params = self.get_predictions()
        
        # Cria o DataFrame esperado pelos novos plots
        df = pd.DataFrame({
            'Price_Real': y_real,
            'Price_Pred': y_pred,
            'S': S,
            'K': K,
            'T': T
        })
        
        # Métricas derivadas
        df['Moneyness'] = df['S'] / (df['K'] + 1e-8)
        df['Error'] = df['Price_Real'] - df['Price_Pred']
        df['Abs_Error'] = np.abs(df['Error'])
        
        # Adiciona parâmetros Heston
        param_names = ['Heston_nu', 'Heston_theta', 'Heston_kappa', 'Heston_xi', 'Heston_rho']
        # Garante que temos colunas suficientes
        num_params = min(params.shape[1], len(param_names))
        for i in range(num_params):
            df[param_names[i]] = params[:, i]
            
        return df
    # ==========================================================================
    # SEÇÃO 3: Validação Física - Histórico de Treinamento
    # ==========================================================================
    
    def plot_loss_history(self):
        """
        Plot de convergência das componentes de perda.
        FOCO: Resíduo PDE, Boundary Conditions, Data Loss
        """
        if self.history is None or self.history.empty:
            logger.warning("Histórico vazio. Pulando plot_loss_history.")
            return
        
        try:
            # Criar subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Perda Total (Log Scale)',
                    'Resíduo da EDP (PDE Loss)',
                    'Condições de Contorno (BC Loss)',
                    'Erro de Dados (Data Loss)'
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.10
            )
            
            # Criar epochs a partir do índice se não existir coluna 'epoch'
            if 'epoch' in self.history.columns:
                epochs = self.history['epoch'].values
            else:
                epochs = np.arange(len(self.history))
            
            # 1. Perda Total (usar train_loss se total_loss não existir)
            loss_col = 'total_loss' if 'total_loss' in self.history.columns else 'train_loss'
            if loss_col in self.history.columns:
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=self.history[loss_col],
                        mode='lines', name='Total Loss',
                        line=dict(color='#2E86AB', width=2)
                    ),
                    row=1, col=1
                )
                fig.update_yaxes(type="log", row=1, col=1)
            
            # 2. Resíduo PDE (MÉTRICA FUNDAMENTAL DE FÍSICA)
            pde_col = 'pde_loss' if 'pde_loss' in self.history.columns else 'loss_pde'
            if pde_col in self.history.columns:
                if pde_col in self.history.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=epochs, y=self.history[pde_col],
                            mode='lines', name='PDE Residual',
                            line=dict(color='#A23B72', width=2)
                        ),
                        row=1, col=2
                    )
                
                # Zona de convergência física
                fig.add_hline(
                    y=1e-3, line_dash="dash", line_color="orange",
                    annotation_text="Limiar Físico",
                    row=1, col=2
                )
                fig.add_hrect(
                    y0=1e-4, y1=1e-3,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                    row=1, col=2
                )
                fig.update_yaxes(type="log", row=1, col=2)
            
            # 3. Boundary Conditions (usar loss_bc se disponível)
            bc_col = 'bc_loss' if 'bc_loss' in self.history.columns else 'loss_bc'
            if bc_col in self.history.columns:
                if bc_col in self.history.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=epochs, y=self.history[bc_col],
                            mode='lines', name='BC Loss',
                            line=dict(color='#F18F01', width=2)
                        ),
                        row=2, col=1
                    )
                    fig.update_yaxes(type="log", row=2, col=1)
            
            # 4. Data Loss
            data_col = 'data_loss' if 'data_loss' in self.history.columns else 'loss_data'
            if data_col in self.history.columns:
                if data_col in self.history.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=epochs, y=self.history[data_col],
                            mode='lines', name='Data Loss',
                            line=dict(color='#0077B6', width=2)
                        ),
                        row=2, col=2
                    )
                    fig.update_yaxes(type="log", row=2, col=2)
            
            # Layout
            fig.update_layout(
                title_text="Validação Física: Convergência das Componentes de Perda",
                title_font_size=16,
                showlegend=True,
                height=700,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Época", row=2, col=1)
            fig.update_xaxes(title_text="Época", row=2, col=2)
            
            # Salvar
            output_path = os.path.join(self.save_dir, 'loss_convergence.html')
            fig.write_html(output_path)
            logger.info(f"✓ Loss history salvo: {output_path}")
            
            if self.config.get('export_plots_png', False):
                try:
                    fig.write_image(output_path.replace('.html', '.png'), width=1400, height=700)
                except Exception as e:
                    logger.warning(f"Falha ao exportar PNG (kaleido necessário): {e}")
                    
            return fig
        
        except Exception as e:
            logger.error(f"Erro em plot_loss_history: {e}")
            
    def plot_loss_convergence_hybrid(self):
        """
        Alias para plot_loss_history com suporte a subplots específicos de PINN.
        """
        return self.plot_loss_history()

    def plot_overfitting_detection(self):
        """
        Gera gráfico comparativo entre erro de treino e validação.
        """
        if self.history is None or self.history.empty:
            return None
            
        try:
            fig = go.Figure()
            epochs = np.arange(len(self.history))
            
            # Treino vs Validação
            if 'train_loss' in self.history.columns and 'val_loss' in self.history.columns:
                fig.add_trace(go.Scatter(x=epochs, y=self.history['train_loss'], name='Train Loss'))
                fig.add_trace(go.Scatter(x=epochs, y=self.history['val_loss'], name='Val Loss'))
            
            fig.update_layout(
                title="Detecção de Overfitting: Train vs Val Loss",
                xaxis_title="Época",
                yaxis_title="Loss",
                yaxis_type="log",
                template='plotly_white'
            )
            
            output_path = os.path.join(self.save_dir, 'overfitting_detection.html')
            fig.write_html(output_path)
            return fig
        except Exception as e:
            logger.error(f"Erro em plot_overfitting_detection: {e}")
            return None
    
    def plot_weights_history(self):
        """
        Evolução dos pesos adaptativos (se usado).
        Silenciosamente pula se não houver pesos adaptativos no histórico.
        """
        if self.history is None:
            return
        
        # Procurar por colunas de pesos com diferentes nomes possíveis
        weight_data_cols = [col for col in self.history.columns if 'weight' in col.lower() and 'data' in col.lower()]
        weight_pde_cols = [col for col in self.history.columns if 'weight' in col.lower() and ('pde' in col.lower() or 'phy' in col.lower())]
        
        # Se não encontrar nenhuma coluna de peso, retorna silenciosamente
        if not weight_data_cols and not weight_pde_cols:
            logger.debug("Pesos adaptativos não encontrados no histórico. Pulando plot_weights_history.")
            return
        
        try:
            fig = go.Figure()
            
            # Criar epochs a partir do índice se não existir coluna 'epoch'
            if 'epoch' in self.history.columns:
                epochs = self.history['epoch'].values
            else:
                epochs = np.arange(len(self.history))
            
            # Adicionar traço para peso de dados (primeira coluna encontrada)
            if weight_data_cols:
                data_col = weight_data_cols[0]
                fig.add_trace(go.Scatter(
                    x=epochs, y=self.history[data_col],
                    mode='lines', name='Peso Dados',
                    line=dict(color='#0077B6', width=2)
                ))
            
            # Adicionar traço para peso de física/PDE (primeira coluna encontrada)
            if weight_pde_cols:
                pde_col = weight_pde_cols[0]
                fig.add_trace(go.Scatter(
                    x=epochs, y=self.history[pde_col],
                    mode='lines', name='Peso PDE',
                    line=dict(color='#A23B72', width=2)
                ))
            
            fig.update_layout(
                title="Evolução dos Pesos Adaptativos (Data vs Física)",
                xaxis_title="Época",
                yaxis_title="Peso",
                template='plotly_white',
                height=500
            )
            
            output_path = os.path.join(self.save_dir, 'weights_history.html')
            fig.write_html(output_path)
            logger.info(f"✓ Weights history salvo: {output_path}")
        
        except Exception as e:
            logger.error(f"Erro em plot_weights_history: {e}")
    
    def _save_figure(self, fig, filename: str):
        """
        Helper para salvar figuras com nomenclatura consistente.
        
        Args:
            fig: Objeto plotly Figure
            filename: Nome do arquivo sem extensão
        """
        try:
            output_path = os.path.join(self.save_dir, f'{filename}.html')
            fig.write_html(output_path)
            logger.info(f"✓ Plot salvo: {output_path}")
            
            # Exportar PNG se configurado
            if self.config.get('export_plots_png', False):
                try:
                    png_path = output_path.replace('.html', '.png')
                    fig.write_image(png_path, width=1400, height=700)
                except Exception as e:
                    logger.debug(f"PNG export falhou (kaleido necessário): {e}")
        except Exception as e:
            logger.error(f"Erro ao salvar figura {filename}: {e}")
    
    # ==========================================================================
    # SEÇÃO 4: Validação de Precificação
    # ==========================================================================
    
    def _extract_validation_predictions(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Extrai predições do modelo no conjunto de validação.
        
        Returns:
            (y_true, y_pred, metadata_dict)
        """
        y_true_list = []
        y_pred_list = []
        metadata = {'S': [], 'K': [], 'T': [], 'moneyness': []}
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                # Desempacotar batch - CORRIGIDO: 3 valores (sem asset_ids)
                if len(batch) == 4:
                    seq_data, x_phy, target_price, asset_ids = batch
                else:
                    # Caso tenha apenas 3 valores
                    seq_data, x_phy, target_price = batch
                    asset_ids = torch.zeros(len(target_price), dtype=torch.long).to(self.device)
                
                seq_data = seq_data.to(self.device)
                x_phy = x_phy.to(self.device)
                asset_ids = asset_ids.to(self.device)
                
                # Forward pass
                output = self.model(seq_data, x_phy, asset_ids)
                pred_price = output['price']
                
                # Desnormalizar
                pred_price_denorm = pred_price * self.data_stats['P_std'] + self.data_stats['P_mean']
                target_denorm = target_price * self.data_stats['P_std'] + self.data_stats['P_mean']
                
                y_true_list.append(target_denorm.cpu().numpy())
                y_pred_list.append(pred_price_denorm.cpu().numpy())
                
                # Metadados
                S_denorm = x_phy[:, 0] * self.data_stats['S_std'] + self.data_stats['S_mean']
                K_denorm = x_phy[:, 1] * self.data_stats['K_std'] + self.data_stats['K_mean']
                T_denorm = x_phy[:, 2] * self.data_stats['T_std'] + self.data_stats['T_mean']
                
                metadata['S'].append(S_denorm.cpu().numpy())
                metadata['K'].append(K_denorm.cpu().numpy())
                metadata['T'].append(T_denorm.cpu().numpy())
                metadata['moneyness'].append((S_denorm / K_denorm).cpu().numpy())
        
        y_true = np.concatenate(y_true_list).flatten()
        y_pred = np.concatenate(y_pred_list).flatten()
        
        for key in metadata:
            metadata[key] = np.concatenate(metadata[key]).flatten()
        
        return y_true, y_pred, metadata
    
    def plot_prediction_scatter(self):
        """
        Scatter plot: Previsões vs Realidade
        Com linha de identidade e métricas de erro
        """
        try:
            # Usar _run_inference ao invés de _extract_validation_predictions
            data = self._run_inference()
            y_true = data['price_real'].flatten()
            y_pred = data['price_pred'].flatten()
            
            # Calcular moneyness
            S = data['inputs_phy'][:, 0].flatten()
            K = data['inputs_phy'][:, 1].flatten()
            moneyness = S / (K + 1e-8)
            
            # Calcular métricas
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            
            fig = go.Figure()
            
            # Scatter
            fig.add_trace(go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                marker=dict(
                    size=4,
                    color=moneyness,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Moneyness")
                ),
                name='Predições',
                text=[f"M: {m:.2f}" for m in moneyness],
                hovertemplate='Real: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}<extra></extra>'
            ))
            
            # Linha de identidade
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Ideal (y=x)'
            ))
            
            # Anotação com métricas
            annotation_text = (
                f"<b>Métricas de Validação</b><br>"
                f"MAE: {mae:.4f}<br>"
                f"RMSE: {rmse:.4f}<br>"
                f"MAPE: {mape:.2f}%<br>"
                f"R²: {r2:.4f}"
            )
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=annotation_text,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=11),
                align="left",
                xanchor="left", yanchor="top"
            )
            
            fig.update_layout(
                title="Validação de Precificação: Predições vs Ground Truth",
                xaxis_title="Preço Real",
                yaxis_title="Preço Predito",
                template='plotly_white',
                height=600,
                width=800
            )
            
            output_path = os.path.join(self.save_dir, 'prediction_scatter.html')
            fig.write_html(output_path)
            logger.info(f"✓ Prediction scatter salvo: {output_path}")
            logger.info(f"  Métricas: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        except Exception as e:
            logger.error(f"Erro em plot_prediction_scatter: {e}")
    
    def plot_residuals(self):
        """
        Análise de resíduos (erros de predição).
        Histograma e scatter vs moneyness.
        """
        try:
            # Usar _run_inference
            data = self._run_inference()
            y_true = data['price_real'].flatten()
            y_pred = data['price_pred'].flatten()
            residuals = y_true - y_pred
            
            # Calcular moneyness
            S = data['inputs_phy'][:, 0].flatten()
            K = data['inputs_phy'][:, 1].flatten()
            moneyness = S / (K + 1e-8)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Distribuição dos Resíduos', 'Resíduos vs Moneyness')
            )
            
            # Histograma
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=50,
                    name='Resíduos',
                    marker_color='#0077B6',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Scatter: Resíduos vs Moneyness
            fig.add_trace(
                go.Scatter(
                    x=moneyness,
                    y=residuals,
                    mode='markers',
                    marker=dict(size=4, color='#A23B72', opacity=0.5),
                    name='Resíduos'
                ),
                row=1, col=2
            )
            
            # Linha zero
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
            
            fig.update_layout(
                title_text="Análise de Resíduos de Precificação",
                showlegend=False,
                height=500,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Resíduo (Real - Pred)", row=1, col=1)
            fig.update_xaxes(title_text="Moneyness (S/K)", row=1, col=2)
            fig.update_yaxes(title_text="Frequência", row=1, col=1)
            fig.update_yaxes(title_text="Resíduo", row=1, col=2)
            
            output_path = os.path.join(self.save_dir, 'residuals_analysis.html')
            fig.write_html(output_path)
            logger.info(f"✓ Residuals analysis salvo: {output_path}")
        
        except Exception as e:
            logger.error(f"Erro em plot_residuals: {e}")
    
    def plot_error_by_moneyness(self):
        """
        Erro médio absoluto em função do moneyness.
        Identifica regiões problemáticas (ITM, ATM, OTM).
        """
        try:
            # Usar _run_inference
            data = self._run_inference()
            y_true = data['price_real'].flatten()
            y_pred = data['price_pred'].flatten()
            errors = np.abs(y_true - y_pred)
            
            # Calcular moneyness
            S = data['inputs_phy'][:, 0].flatten()
            K = data['inputs_phy'][:, 1].flatten()
            moneyness = S / (K + 1e-8)
            
            # Binning por moneyness
            bins = np.linspace(0.7, 1.3, 20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            mean_errors = []
            std_errors = []
            
            for i in range(len(bins) - 1):
                mask = (moneyness >= bins[i]) & (moneyness < bins[i+1])
                if mask.sum() > 0:
                    mean_errors.append(errors[mask].mean())
                    std_errors.append(errors[mask].std())
                else:
                    mean_errors.append(0)
                    std_errors.append(0)
            
            mean_errors = np.array(mean_errors)
            std_errors = np.array(std_errors)
            
            fig = go.Figure()
            
            # Linha com banda de erro
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=mean_errors,
                mode='lines+markers',
                name='MAE Médio',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=8)
            ))
            
            # Banda de desvio padrão
            fig.add_trace(go.Scatter(
                x=np.concatenate([bin_centers, bin_centers[::-1]]),
                y=np.concatenate([mean_errors + std_errors, (mean_errors - std_errors)[::-1]]),
                fill='toself',
                fillcolor='rgba(46, 134, 171, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Desvio Padrão',
                showlegend=True
            ))
            
            # Zonas de moneyness
            fig.add_vrect(x0=0.7, x1=0.95, fillcolor="red", opacity=0.05, 
                         annotation_text="OTM", annotation_position="top left")
            fig.add_vrect(x0=0.95, x1=1.05, fillcolor="green", opacity=0.05,
                         annotation_text="ATM", annotation_position="top")
            fig.add_vrect(x0=1.05, x1=1.3, fillcolor="blue", opacity=0.05,
                         annotation_text="ITM", annotation_position="top right")
            
            fig.update_layout(
                title="Erro de Precificação por Moneyness",
                xaxis_title="Moneyness (S/K)",
                yaxis_title="MAE Médio",
                template='plotly_white',
                height=500
            )
            
            output_path = os.path.join(self.save_dir, 'error_by_moneyness.html')
            fig.write_html(output_path)
            logger.info(f"✓ Error by moneyness salvo: {output_path}")
        
        except Exception as e:
            logger.error(f"Erro em plot_error_by_moneyness: {e}")
    
    # ==========================================================================
    # SEÇÃO 5: Validação Física - Superfícies e Gregas
    # ==========================================================================
    
    def _generate_synthetic_grid(self, n_points: int = 40) -> Tuple:
        """
        Gera grid sintético para visualização de superfícies.
        Baseado no visualization_old.py - usa amostra real para inicializar LSTM.
        
        Args:
            n_points: Resolução do grid
        
        Returns:
            (x_phy, heston_params, S_grid, T_grid, asset_ids)
        """
        try:
            # Pega uma amostra real para gerar o estado inicial da LSTM
            try:
                sample_batch = next(iter(self.val_loader))
            except StopIteration:
                logger.error("Dataloader vazio. Não é possível gerar grid sintético.")
                return None, None, None, None, None

            x_seq_sample = sample_batch[0].to(self.device)
            # asset_ids pode estar na posição 3 (se batch = 4) ou 5 (se batch ≥ 6)
            if len(sample_batch) >= 6:
                asset_ids_sample = sample_batch[5].to(self.device)
            elif len(sample_batch) == 4:
                asset_ids_sample = sample_batch[3].to(self.device)
            else:
                asset_ids_sample = torch.zeros(x_seq_sample.size(0), dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                # Preparar input da LSTM com embeddings (se modelo usar)
                if hasattr(self.model, 'use_embedding') and self.model.use_embedding:
                    # [Batch, Emb_Dim]
                    emb = self.model.asset_embedding(asset_ids_sample)
                    # [Batch, Seq, Emb_Dim]
                    emb_seq = emb.unsqueeze(1).repeat(1, x_seq_sample.size(1), 1)
                    # [Batch, Seq, Features+Emb]
                    lstm_input = torch.cat([x_seq_sample, emb_seq], dim=2)
                else:
                    lstm_input = x_seq_sample

                # Chama LSTM para obter estados ocultos
                _, (h_n, _) = self.model.lstm(lstm_input)
                
                # Tira a média do estado de mercado para ter um "regime médio"
                market_state_avg = h_n[-1].mean(dim=0, keepdim=True)
                market_state_expanded = market_state_avg.repeat(n_points*n_points, 1)
                
                # Gera parâmetros Heston baseados nesse regime médio
                nu, theta, kappa, xi, rho = self.model.heston_head(market_state_expanded)

            # Grid físico
            s_vals = np.linspace(0.6, 1.4, n_points)  # Moneyness range
            t_vals = np.linspace(0.1, 1.0, n_points)  # Time range (anos)
            S_grid, T_grid = np.meshgrid(s_vals, t_vals)
            
            # Converte para tensores [N*N, 1]
            S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=self.device).unsqueeze(1)
            T_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=self.device).unsqueeze(1)
            K_flat = torch.ones_like(S_flat, device=self.device) * 1.0   # Strike normalizado
            r_flat = torch.ones_like(S_flat, device=self.device) * 0.10  # Juros fixos 10%
            q_flat = torch.ones_like(S_flat, device=self.device) * 0.02  # Dividend yield fixo 2%
            
            # Input físico: [S, K, T, r, q]
            x_phy_synthetic = torch.cat([S_flat, K_flat, T_flat, r_flat, q_flat], dim=1).to(self.device).requires_grad_(True)
            
            # Asset ID Fictício
            asset_ids_synthetic = torch.zeros(x_phy_synthetic.shape[0], dtype=torch.long, device=self.device)

            # Parâmetros Heston (com gradiente habilitado para a EDP)
            fixed_heston_params = (
                nu.detach().to(self.device).requires_grad_(True),
                theta.detach().to(self.device).requires_grad_(True),
                kappa.detach().to(self.device).requires_grad_(True),
                xi.detach().to(self.device).requires_grad_(True),
                rho.detach().to(self.device).requires_grad_(True)
            )
            
            return x_phy_synthetic, fixed_heston_params, S_grid, T_grid, asset_ids_synthetic
        
        except Exception as e:
            logger.error(f"Erro ao gerar grid sintético: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None, None
    
    def plot_price_surface(self):
        """
        Superfície 3D: Preço da opção em função de (S, T).
        Valida se a superfície é suave e fisicamente consistente.
        """
        try:
            x_phy, heston_params, S_grid, T_grid, asset_ids = self._generate_synthetic_grid(n_points=40)
            
            if x_phy is None:
                logger.warning("Grid sintético falhou. Pulando price_surface.")
                return
            
            # Desempacotar parâmetros Heston
            nu, theta, kappa, xi, rho = heston_params
            pinn_input = torch.cat([x_phy, nu, theta, kappa, xi, rho], dim=1)
            
            # Projeção Fourier (se o modelo usar)
            if hasattr(self.model, 'use_fourier') and self.model.use_fourier:
                feats = self.model.fourier_layer(pinn_input)
            else:
                feats = pinn_input
            
            # Forward pass
            with torch.no_grad():
                nn_out = self.model.pricing_net(feats)
                
                # Aplicar Hard Constraint manualmente para visualização correta
                T_norm = x_phy[:, 2:3]
                time_factor = 1.0 - torch.exp(-10.0 * T_norm)
                time_value = time_factor * torch.nn.functional.softplus(nn_out)
                
                # Payoff (S e K estão normalizados e alinhados no grid)
                payoff = torch.relu(x_phy[:, 0:1] - x_phy[:, 1:2])
                price = payoff + time_value
            
            # Reshape para grid
            Z = price.cpu().numpy().reshape(S_grid.shape)
            
            # Desnormalizar eixos
            S_denorm = S_grid * self.data_stats['S_std'] + self.data_stats['S_mean']
            T_denorm = T_grid * self.data_stats['T_std'] + self.data_stats['T_mean']
            
            # Plot 3D
            fig = go.Figure(data=[go.Surface(
                x=S_denorm,
                y=T_denorm,
                z=Z,
                colorscale='Viridis',
                name='Preço'
            )])
            
            fig.update_layout(
                title="Superfície de Preço: V(S, τ)",
                scene=dict(
                    xaxis_title="Spot Price (S)",
                    yaxis_title="Tempo até Vencimento (τ)",
                    zaxis_title="Preço Normalizado (P/K)"
                ),
                height=700,
                template='plotly_white'
            )
            
            output_path = os.path.join(self.save_dir, 'price_surface_3d.html')
            fig.write_html(output_path)
            logger.info(f"✓ Price surface 3D salvo: {output_path}")
        
        except Exception as e:
            logger.error(f"Erro em plot_price_surface: {e}")
    
    def plot_delta_surface(self):
        """
        Superfície 3D: Delta (∂V/∂S).
        Valida se o Delta é suave e sem oscilações não físicas.
        """
        try:
            x_phy, heston_params, S_grid, T_grid, asset_ids = self._generate_synthetic_grid(n_points=40)
            
            if x_phy is None:
                return
            
            x_phy.requires_grad_(True)
            
            # Desempacotar parâmetros Heston
            nu, theta, kappa, xi, rho = heston_params
            pinn_input = torch.cat([x_phy, nu, theta, kappa, xi, rho], dim=1)
            
            # Projeção Fourier (se o modelo usar)
            if hasattr(self.model, 'use_fourier') and self.model.use_fourier:
                feats = self.model.fourier_layer(pinn_input)
            else:
                feats = pinn_input
            
            # Forward pass para obter preço
            price = torch.nn.functional.softplus(self.model.pricing_net(feats))
            
            # Calcular Delta via autodiff - Gradiente dV/dS (S está no índice 0 de x_phy)
            grads = torch.autograd.grad(
                outputs=price,
                inputs=x_phy,
                grad_outputs=torch.ones_like(price),
                create_graph=False,
                retain_graph=False
            )[0]
            delta = grads[:, 0].cpu().numpy()
            
            # Reshape para grid
            Z_delta = delta.reshape(S_grid.shape)
            
            # Desnormalizar eixos
            S_denorm = S_grid * self.data_stats['S_std'] + self.data_stats['S_mean']
            T_denorm = T_grid * self.data_stats['T_std'] + self.data_stats['T_mean']
            
            # Plot 3D
            fig = go.Figure(data=[go.Surface(
                x=S_denorm,
                y=T_denorm,
                z=Z_delta,
                colorscale='RdBu',
                name='Delta'
            )])
            
            fig.update_layout(
                title="Superfície de Delta: ∂V/∂S",
                scene=dict(
                    xaxis_title="Spot Price (S)",
                    yaxis_title="Tempo até Vencimento (τ)",
                    zaxis_title="Delta"
                ),
                height=700,
                template='plotly_white'
            )
            
            output_path = os.path.join(self.save_dir, 'delta_surface_3d.html')
            fig.write_html(output_path)
            logger.info(f"✓ Delta surface 3D salvo: {output_path}")
        
        except Exception as e:
            logger.error(f"Erro em plot_delta_surface: {e}")
    
    def plot_pde_residual_surface(self):
        """
        Superfície 3D: Resíduo da EDP.
        MÉTRICA CRÍTICA: Deve ser próximo de zero em todo o domínio.
        """
        try:
            x_phy, heston_params, S_grid, T_grid, asset_ids = self._generate_synthetic_grid(n_points=40)
            
            if x_phy is None:
                return
            
            x_phy.requires_grad_(True)
            
            # Desempacotar parâmetros Heston
            nu, theta, kappa, xi, rho = heston_params
            pinn_input = torch.cat([x_phy, nu, theta, kappa, xi, rho], dim=1)
            
            # Projeção Fourier (se o modelo usar)
            if hasattr(self.model, 'use_fourier') and self.model.use_fourier:
                feats = self.model.fourier_layer(pinn_input)
            else:
                feats = pinn_input
            
            # Forward pass com gradientes ativos
            nn_out = self.model.pricing_net(feats)
            
            # Hard Constraint (Reimplementação manual para derivar)
            T_norm = x_phy[:, 2:3]
            time_factor = 1.0 - torch.exp(-10.0 * T_norm)
            time_value = time_factor * torch.nn.functional.softplus(nn_out)
            
            # Payoff (S e K estão normalizados no grid)
            payoff = torch.relu(x_phy[:, 0:1] - x_phy[:, 1:2])
            price = payoff + time_value
            
            # Calcular resíduo PDE
            model_output = {'price': price, 'heston_params': heston_params}
            res = heston_residual(model_output, x_phy, self.data_stats, return_residuals=True)
            
            # Resíduo quadrático
            Z_res = (res**2).detach().cpu().numpy().reshape(S_grid.shape)
            
            # Verificação de NaNs
            if np.isnan(Z_res).any() or np.isinf(Z_res).any():
                logger.warning("NaNs ou Infs detectados na superfície de resíduo. Substituindo por 0 para plot.")
                Z_res = np.nan_to_num(Z_res, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Valor médio do resíduo para exibição
            mean_residual = np.mean(Z_res)
            
            # Desnormalizar eixos
            S_denorm = S_grid * self.data_stats['S_std'] + self.data_stats['S_mean']
            T_denorm = T_grid * self.data_stats['T_std'] + self.data_stats['T_mean']
            
            # Plot 3D
            fig = go.Figure(data=[go.Surface(
                x=S_denorm,
                y=T_denorm,
                z=Z_res,
                colorscale='Hot',
                name='Resíduo PDE'
            )])
            
            fig.update_layout(
                title=f"Resíduo da EDP (Valor Médio: {mean_residual:.2e})",
                scene=dict(
                    xaxis_title="Spot Price (S)",
                    yaxis_title="Tempo até Vencimento (τ)",
                    zaxis_title="|Resíduo PDE|²"
                ),
                height=700,
                template='plotly_white'
            )
            
            output_path = os.path.join(self.save_dir, 'pde_residual_surface.html')
            fig.write_html(output_path)
            logger.info(f"✓ PDE residual surface salvo: {output_path}")
            logger.info(f"  Resíduo PDE médio: {mean_residual:.2e}")
        
        except Exception as e:
            logger.error(f"Erro em plot_pde_residual_surface: {e}")
    
    # ==========================================================================
    # SEÇÃO 6: Validação LSTM - Parâmetros Heston
    # ==========================================================================
    
    def plot_heston_params(self):
        """
        Distribuição dos parâmetros Heston previstos pela LSTM.
        Valida se estão em ranges realistas.
        """
        try:
            # Usar cache de _run_inference
            data = self._run_inference()
            
            if data['heston_params'] is None:
                logger.warning("Parâmetros Heston não disponíveis no cache.")
                return
            
            params = data['heston_params']  # Shape: [N, 5]
            
            # Organizar em dicionário
            params_dict = {
                'nu': params[:, 0],
                'theta': params[:, 1],
                'kappa': params[:, 2],
                'xi': params[:, 3],
                'rho': params[:, 4]
            }
            
            # Subplots
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=list(params_dict.keys())
            )
            
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#0077B6', '#C73E1D']
            
            for (row, col), (name, values), color in zip(positions, params_dict.items(), colors):
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        name=name,
                        marker_color=color,
                        nbinsx=30,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title_text="Distribuição dos Parâmetros Heston (LSTM)",
                showlegend=False,
                height=600,
                template='plotly_white'
            )
            
            output_path = os.path.join(self.save_dir, 'heston_params_distribution.html')
            fig.write_html(output_path)
            logger.info(f"✓ Heston params distribution salvo: {output_path}")
            
            # Log estatísticas
            for name, values in params_dict.items():
                logger.info(f"  {name}: mean={values.mean():.4f}, std={values.std():.4f}")
        
        except Exception as e:
            logger.error(f"Erro em plot_heston_params: {e}")
    
    # ==========================================================================
    # SEÇÃO 7: Plots Adicionais (Compatibilidade)
    # ==========================================================================
    
    def plot_premium_over_time(self):
        """
        Visualiza a evolução do prêmio da opção em relação ao Tempo para Vencimento (T).
        Validação Física: Verifica o 'Theta Decay' (o preço deve convergir para o Payoff quando T -> 0).
        """
        try:
            df = self.run_inference()
            
            # Amostragem para performance
            plot_df = df.sample(n=min(2000, len(df))) if len(df) > 2000 else df
            
            fig = go.Figure()
            
            # Scatter colorido por Moneyness para dar contexto
            fig.add_trace(go.Scatter(
                x=plot_df['T'],
                y=plot_df['Price_Pred'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=plot_df['Moneyness'],
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(title="Moneyness (S/K)")
                ),
                text=plot_df.apply(lambda row: f"S/K: {row['Moneyness']:.2f}", axis=1),
                name='Preço PINN'
            ))
            
            fig.update_layout(
                title="Prêmio da Opção vs. Tempo para Vencimento (Theta Decay)",
                xaxis_title="Tempo para Vencimento (Anos)",
                yaxis_title="Preço da Opção (Normalizado)",
                hovermode="closest",
                height=600
            )
            self._save_figure(fig, '3d_premium_over_time')
            
        except Exception as e:
            logger.error(f"Erro em plot_premium_over_time: {e}")
    
    def plot_model_vs_bs_comparison(self):
        """
        Comparativo Preço de Mercado (Real) vs Modelo Híbrido.
        Gráfico de Paridade (QQ Plot).
        """
        try:
            df = self.run_inference()
            
            # Amostragem
            plot_df = df.sample(n=min(3000, len(df))) if len(df) > 3000 else df
            
            # Calcular métricas diretamente
            y_true = plot_df['Price_Real'].values
            y_pred = plot_df['Price_Pred'].values
            metrics_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            metrics_r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            
            fig = go.Figure()
            
            # Pontos do Modelo
            fig.add_trace(go.Scattergl(
                x=plot_df['Price_Real'],
                y=plot_df['Price_Pred'],
                mode='markers',
                marker=dict(color='blue', size=3, opacity=0.5),
                name=f'PINN (R²={metrics_r2:.4f})'
            ))
            
            # Linha de Identidade (Perfeição)
            max_val = max(plot_df['Price_Real'].max(), plot_df['Price_Pred'].max())
            min_val = min(plot_df['Price_Real'].min(), plot_df['Price_Pred'].min())
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Identidade (y=x)'
            ))
            
            fig.update_layout(
                title=f"Aderência do Modelo: PINN vs Mercado (RMSE: {metrics_rmse:.4f})",
                xaxis_title="Preço de Mercado",
                yaxis_title="Preço Previsto (PINN)",
                height=600,
                width=800,
                template='plotly_white'
            )
            self._save_figure(fig, '3e_model_vs_market_comparison')
            
        except Exception as e:
            logger.error(f"Erro em plot_model_vs_bs_comparison: {e}")
    
    def plot_distribution_overlay(self):
        """
        Sobreposição de histogramas: Distribuição dos Preços Reais vs Previstos.
        Verifica se o modelo captura a variância e a cauda da distribuição real.
        """
        try:
            df = self.run_inference()
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df['Price_Real'],
                name='Mercado (Real)',
                opacity=0.6,
                marker_color='green',
                histnorm='probability density'
            ))
            
            fig.add_trace(go.Histogram(
                x=df['Price_Pred'],
                name='Modelo (Previsto)',
                opacity=0.6,
                marker_color='blue',
                histnorm='probability density'
            ))
            
            fig.update_layout(
                title="Densidade de Distribuição de Preços",
                xaxis_title="Preço da Opção",
                yaxis_title="Densidade",
                barmode='overlay',
                height=500
            )
            self._save_figure(fig, '3f_distribution_overlay')
            
        except Exception as e:
            logger.error(f"Erro em plot_distribution_overlay: {e}")
    
    def plot_error_distribution(self):
        """Histograma dos Resíduos (Erro)."""
        try:
            df = self.run_inference()
            mu = df['Error'].mean()
            sigma = df['Error'].std()
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['Error'],
                nbinsx=100,
                marker_color='purple',
                name='Resíduos'
            ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            fig.add_annotation(x=mu, y=0, text=f"Mean: {mu:.4f}", showarrow=True, arrowhead=1)
            
            fig.update_layout(title=f"Distribuição de Erros (Std Dev: {sigma:.4f})", xaxis_title="Erro (Real - Previsto)")
            self._save_figure(fig, '3g_error_distribution')
        except Exception as e:
            logger.error(f"Erro em plot_error_distribution: {e}")

    def plot_error_heatmap(self):
        """
        Heatmap de Erro Absoluto Médio (MAE) segmentado por Moneyness e Tempo.
        Identifica 'Zonas Cegas' do modelo (ex: Opções Deep OTM perto do vencimento).
        """
        try:
            df = self.run_inference()
            
            # Cria bins para Moneyness e Tempo
            df['M_bin'] = pd.cut(df['Moneyness'], bins=np.linspace(0.7, 1.3, 11), labels=False)
            df['T_bin'] = pd.cut(df['T'], bins=np.linspace(0, 1.0, 11), labels=False)
            
            # Pivot table para calcular MAE por bin
            pivot_mae = df.pivot_table(values='Abs_Error', index='T_bin', columns='M_bin', aggfunc='mean')
            
            # Labels para eixos
            x_labels = [f"{v:.2f}" for v in np.linspace(0.7, 1.3, 11)]
            y_labels = [f"{v:.2f}" for v in np.linspace(0, 1.0, 11)]
            
            fig = px.imshow(
                pivot_mae,
                labels=dict(x="Moneyness (S/K)", y="Tempo (Anos)", color="MAE"),
                x=x_labels[:-1],
                y=y_labels[:-1],
                color_continuous_scale='Viridis',
                origin='lower'
            )
            
            fig.update_layout(title="Mapa de Calor de Erro (MAE): Onde o modelo erra mais?", height=600)
            self._save_figure(fig, '3h_error_heatmap')
            
        except Exception as e:
            logger.error(f"Erro em plot_error_heatmap: {e}")
    
     
    def plot_vol_smile(self):
        """
        Visualiza o 'Smile' de Volatilidade indiretamente via Preço vs Moneyness.
        Filtra para um corte de tempo específico (ex: T entre 0.1 e 0.2) para clareza.
        """
        try:
            df = self.run_inference()
            
            # Filtra uma 'fatia' de tempo (ex: opções curtas, 30-60 dias)
            # T está em anos. 30/252 ~= 0.12
            mask = (df['T'] > 0.08) & (df['T'] < 0.2)
            slice_df = df[mask]
            
            if len(slice_df) < 50:
                logger.warning("Dados insuficientes para plot de Smile (fatia de tempo muito pequena).")
                return

            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=slice_df['Moneyness'],
                y=slice_df['Price_Real'],
                mode='markers',
                name='Mercado',
                marker=dict(color='green', size=5, symbol='circle-open')
            ))
            
            # Ordena para linha suave
            slice_df_sorted = slice_df.sort_values('Moneyness')
            
            fig.add_trace(go.Scatter(
                x=slice_df_sorted['Moneyness'],
                y=slice_df_sorted['Price_Pred'],
                mode='lines',
                name='PINN Heston',
                line=dict(color='blue', width=3)
            ))
            
            fig.update_layout(
                title="Curva de Preços (Proxy de Smile) - Vencimento Curto (~30-50 dias)",
                xaxis_title="Moneyness (S/K)",
                yaxis_title="Preço da Opção",
                height=500
            )
            self._save_figure(fig, '3i_volatility_smile_proxy')
            
        except Exception as e:
            logger.error(f"Erro em plot_vol_smile: {e}")
    
    def plot_latent_vol_evolution(self):
        """
        Visualiza a série temporal dos parâmetros latentes inferidos pela LSTM.
        Ex: Volatilidade Instantânea (nu), Média de Longo Prazo (theta), etc.
        """
        try:
            df = self.run_inference()
            
            # Identifica colunas Heston
            heston_cols = [c for c in df.columns if c.startswith('Heston_')]
            
            if not heston_cols:
                logger.warning("Nenhuma coluna Heston encontrada no DataFrame.")
                return
                
            # Como não temos a data exata no df de inferência (as vezes), usamos o índice
            # Se 'times' for recuperado no run_inference, melhor ainda.
            x_axis = df.index 
            if 'times' in df.columns: # Se você tiver salvo o tempo real
                 x_axis = df['times'] # Atenção: isso pode ser TTM reverso
                 
            # Plota os 2 principais: Nu (Vol) e Theta (Long Run)
            fig = make_subplots(rows=len(heston_cols), cols=1, shared_xaxes=True, subplot_titles=heston_cols)
            
            # Amostragem ordenada
            sample_df = df.iloc[::max(1, len(df)//1000)] # Max 1000 pontos
            
            for i, col in enumerate(heston_cols):
                fig.add_trace(
                    go.Scatter(y=sample_df[col], mode='lines', name=col),
                    row=i+1, col=1
                )
                
            fig.update_layout(height=250*len(heston_cols), title="Dinâmica dos Parâmetros Heston (Inferência LSTM)")
            self._save_figure(fig, '4a_latent_vol_evolution')
            
        except Exception as e:
            logger.error(f"Erro em plot_latent_vol_evolution: {e}")
    
    def plot_premium_by_moneyness_time(self):
        """
        Superfície 3D de Preços: Moneyness (X) vs Tempo (Y) vs Preço (Z).
        """
        try:
            df = self.run_inference()
            
            # Amostragem para 3D (muitos pontos deixam pesado)
            sample = df.sample(n=min(3000, len(df)))
            
            fig = go.Figure(data=[go.Scatter3d(
                x=sample['Moneyness'],
                y=sample['T'],
                z=sample['Price_Pred'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=sample['Error'], # Colore pelo erro para diagnóstico
                    colorscale='Viridis',
                    colorbar=dict(title="Erro Absoluto"),
                    opacity=0.8
                )
            )])
            
            fig.update_layout(
                title="Superfície de Preços PINN (Cor = Erro)",
                scene=dict(
                    xaxis_title='Moneyness (S/K)',
                    yaxis_title='Tempo (Anos)',
                    zaxis_title='Preço Previsto'
                ),
                height=700
            )
            self._save_figure(fig, '3j_price_surface_3d')
            
        except Exception as e:
            logger.error(f"Erro em plot_premium_by_moneyness_time: {e}")
    
    # ==========================================================================
    # SEÇÃO 8: Função Principal de Execução
    # ==========================================================================
    
    def plot_all(self):
        """
        Gera todos os gráficos habilitados na configuração.
        Esta é a função chamada pelo main.py ao final do pipeline.
        """
        logger.info("=" * 70)
        logger.info(f"Iniciando geração de gráficos em: {self.save_dir}")
        logger.info("=" * 70)
        
        # Lista de jobs (nome_config, função)
        jobs = [
            # Validação Física - Histórico
            ('plot_loss_convergence', self.plot_loss_history),
            ('plot_weights_history', self.plot_weights_history),
            
            # Validação de Precificação
            ('plot_price_scatter', self.plot_prediction_scatter),
            ('plot_residuals', self.plot_residuals),
            ('plot_error_by_moneyness', self.plot_error_by_moneyness),
            
            # Superfícies e Gregas
            ('plot_price_surface', self.plot_price_surface),
            ('plot_delta_surface', self.plot_delta_surface),
            ('plot_pde_residual', self.plot_pde_residual_surface),
            
            # Validação LSTM
            ('plot_heston_params', self.plot_heston_params),
            
            # Placeholders (compatibilidade)
            ('plot_premium_time', self.plot_premium_over_time),
            ('plot_model_vs_bs', self.plot_model_vs_bs_comparison),
            ('plot_dist_overlay', self.plot_distribution_overlay),
            ('plot_error_distribution', self.plot_error_distribution),
            ('plot_error_heatmap', self.plot_error_heatmap),
            ('plot_vol_smile', self.plot_vol_smile),
            ('plot_latent_vol', self.plot_latent_vol_evolution),
            ('plot_premium_by_moneyness_time', self.plot_premium_by_moneyness_time),
        ]
        
        success_count = 0
        failed_count = 0
        
        for name, func in jobs:
            # Verificar se está habilitado na config
            if self.config.get(name, True):
                try:
                    logger.info(f"Gerando: {name}...")
                    func()
                    success_count += 1
                except Exception as e:
                    logger.error(f"Falha ao gerar '{name}': {e}")
                    failed_count += 1
        
        logger.info("=" * 70)
        logger.info("Rotina de visualização finalizada.")
        logger.info(f"✓ Sucesso: {success_count} plots")
        if failed_count > 0:
            logger.warning(f"✗ Falhas: {failed_count} plots")
        logger.info("=" * 70)
