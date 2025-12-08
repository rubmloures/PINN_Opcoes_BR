# /src/trainer.py

import torch
import torch.nn as nn
import numpy as np
import time
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.physics import heston_residual
from src.config import TRAINING_CONFIG, PATHS
from src.utils import salvar_historico_treinamento
from src.logger import get_logger

# Configurar logger
logger = get_logger('PINN_Trainer')

class PINNTrainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, data_stats: dict, config: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_stats = data_stats
        self.config = config
        self.device = torch.device(self.config.get('device', 'cpu'))

        self.model.to(self.device)
        
        # Histórico expandido para diagnóstico
        self.history = {
            'train_loss': [], 
            'val_loss': [],
            'val_mae': [],
            'lr': [], 
            'loss_data': [], 
            'loss_pde': [],
            'loss_bc': [],
            'loss_reg': [],
            'weight_pde_curr': [],
            'bias_pred': [],
            'pde_data_ratio': [],
            'sigma_data': [],
            'sigma_pde': [],
            'avg_pred_vol': [],
            'avg_pred_price': [],
            'avg_real_price': [],
            'learned_bias': [],  # Novo: monitorar bias aprendível
            'moneyness_preservation': []  # Novo: monitorar preservação de moneyness
        }        
        self.best_val_loss = float('inf')

        # --- Pesos Adaptativos (Learnable Uncertainty) ---
        self.use_adaptive_weights = self.config.get('use_adaptive_weights', False)
        if self.use_adaptive_weights:
            # Parâmetros treináveis: log(sigma^2) para Data e PDE
            self.log_vars = nn.Parameter(torch.zeros(2, device=self.device, requires_grad=True))
    
    def denormalize_price(self, y_normalized: torch.Tensor, K_real: torch.Tensor) -> torch.Tensor:
        """
        Desnormaliza preço: y_pred (normalizado por K) -> preço absoluto.
        
        Args:
            y_normalized: Preço normalizado (y/K)
            K_real: Strike price em escala real
            
        Returns:
            Preço em escala absoluta (preço da opção em unidades monetárias)
        """
        # Fórmula: y_denorm = y_norm * K
        # Garante dimensões compatíveis
        if y_normalized.dim() > K_real.dim():
            K_real = K_real.unsqueeze(-1)
        
        y_denorm = y_normalized * K_real
        
        # Garantir valores não-negativos (preços não podem ser negativos)
        y_denorm = torch.clamp(y_denorm, min=0.0)
        
        return y_denorm
    
    def compute_loss(self, batch, weight_pde_curr):
        """Calcula a perda composta (dados + física) com curriculum learning."""
        # Desempacotar batch com validação
        try:
            if len(batch) == 6:
                x_seq, x_phy, y_real, X_time, weights, asset_ids = batch
            else:
                raise ValueError(f"Batch esperado com 6 elementos, recebido {len(batch)}")
        except Exception as e:
            logger.error(f"Erro ao desempacotar batch: {e}")
            raise
        
        x_seq = x_seq.to(self.device)
        x_phy = x_phy.to(self.device)
        y_real = y_real.to(self.device)
        weights = weights.to(self.device)
        
        # ===== VALIDAÇÃO CRÍTICA: ASSET_IDS =====
        if asset_ids is not None:
            asset_ids = asset_ids.to(self.device)
            if self.model.use_embedding and asset_ids.shape[0] != x_seq.shape[0]:
                logger.warning(
                    f"Mismatch dimensão asset_ids ({asset_ids.shape[0]}) vs "
                    f"x_seq ({x_seq.shape[0]}). Ajustando..."
                )
                asset_ids = asset_ids[:x_seq.shape[0]]
        else:
            logger.debug("Asset IDs são None - usando modelo sem embeddings")
            if self.model.use_embedding:
                logger.warning("Modelo configurado para usar embeddings mas asset_ids é None")

        # Habilita gradiente para física
        x_phy.requires_grad_(True)
        
        # Forward Pass
        outputs = self.model(x_seq, x_phy, asset_ids)  
        pred_price = outputs['price']  # Preço normalizado (P/K)
        
        # 1. Loss de Dados (MSE Ponderado) - usa preços normalizados
        loss_data = torch.mean(weights * (pred_price - y_real) ** 2)
        weight_data = self.config['weight_data']
        
        # 2. Loss da PDE (Heston) - também usa preços normalizados
        lambda_bc = self.config.get('lambda_bc', 1.0)
        lambda_reg = self.config.get('lambda_reg', 0.01)
        
        physics_output = heston_residual(
            outputs, 
            x_phy, 
            self.data_stats,
            lambda_bc=lambda_bc,
            lambda_reg=lambda_reg
        )
        
        # Extrair componentes de physics (sempre retorna dicionário)
        if not isinstance(physics_output, dict):
            raise ValueError(
                f"physics_output deve ser dict, recebido {type(physics_output)}. "
                f"Verifique heston_residual() em src/physics.py"
            )
        
        loss_pde_total = physics_output['total']
        loss_pde = physics_output['pde']
        loss_bc = physics_output.get('bc', torch.tensor(0.0, device=self.device))
        loss_reg = physics_output.get('reg', torch.tensor(0.0, device=self.device))
        
        # 3. Regularização do Bias (evita que fique muito grande)
        bias_reg = 0.01 * (self.model.price_bias ** 2)
        
        # 4. Combinação com Curriculum Learning
        if self.use_adaptive_weights:
            # Pesos adaptativos (Kendall et al.)
            precision_data = torch.exp(-self.log_vars[0])
            precision_pde = torch.exp(-self.log_vars[1])
            
            total_loss = (precision_data * loss_data + self.log_vars[0]) + \
                         (precision_pde * loss_pde_total + self.log_vars[1]) + bias_reg
        else:
            # Pesos fixos com curriculum learning
            weight_data = self.config.get('weight_data')
            total_loss = weight_data * loss_data + weight_pde_curr * loss_pde_total + bias_reg

        return total_loss, loss_data, loss_pde_total, outputs

    def train(self):
        logger.info(f"Iniciando Treinamento Híbrido no dispositivo: {self.device}")
        total_start_time = time.time()
        global_epoch = 0
        
        # Loop de Fases (Curriculum Learning)
        learning_rates = self.config.get('learning_rates', [1e-3, 1e-4, 1e-5])
        
        for phase, lr in enumerate(learning_rates):
            logger.info(f"Fase {phase+1}/{len(learning_rates)} | Learning Rate: {lr}")
            
            # Reinicializa otimizador para a nova fase
            params = list(self.model.parameters())
            if self.use_adaptive_weights:
                params.append(self.log_vars)
            
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
            
            # Reset de variáveis de Early Stopping
            epochs_no_improve = 0
            best_phase_loss = float('inf')
            
            pbar = tqdm(range(self.config.get('epochs_per_phase', 100)), desc=f"Fase {phase+1}")
            
            for epoch in pbar:
                # Curriculum Learning
                global_epoch += 1                
                if global_epoch < self.config['warmup_epochs']:
                    w_pde = 0.0
                elif global_epoch < self.config['warmup_epochs'] + self.config['rampup_epochs']:
                    progress = (global_epoch - self.config['warmup_epochs']) / self.config['rampup_epochs']
                    w_pde = self.config['weight_pde'] * progress
                else:
                    w_pde = self.config['weight_pde']
                    
                self.model.train()
                train_loss_acc = 0
                data_loss_acc = 0
                pde_loss_acc = 0
                
                # Acumuladores para métricas
                pred_vol_acc = 0
                pred_price_acc = 0
                real_price_acc = 0
                num_samples_epoch = 0
                bc_loss_acc = 0
                reg_loss_acc = 0

                # Loop de Batches
                for batch in self.train_loader:
                    self.optimizer.zero_grad()
                    
                    loss, l_data, l_pde, outputs = self.compute_loss(batch, w_pde)
                    
                    if torch.isnan(loss) or torch.isinf(loss) or l_pde > 1e6:
                        # Pula batch corrompido
                        continue
                    
                    loss.backward()
                    
                    # Clip gradiente para estabilidade
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    train_loss_acc += loss.item()
                    data_loss_acc += l_data.item()
                    pde_loss_acc += l_pde.item()
                    
                    # Coleta métricas
                    # CORRIGIDO: Verificação de 'heston_params' antes de acessar
                    if 'heston_params' in outputs and outputs['heston_params'] is not None:
                        nu = outputs['heston_params'][0]
                        pred_vol = torch.sqrt(torch.clamp(nu, min=1e-6)).mean().item()
                    else:
                        pred_vol = 0.0  # Fallback se não houver parâmetros Heston
                    
                    pred_price = outputs['price'].mean().item()
                    
                    # y_real está no batch[2] (Normalizado P/K)
                    y_real_batch = batch[2].to(self.device)
                    x_phy_batch = batch[1].to(self.device)
                    
                    # Desnormaliza y_real para logar o preço real em R$ (apenas para métricas)
                    K_norm_batch = x_phy_batch[:, 1:2]
                    # CORRECTED: Usar Z-score desnormalização (consistente com data_loader)
                    K_real_batch = K_norm_batch * self.data_stats['K_std'] + self.data_stats['K_mean']
                    y_real_denorm = y_real_batch * K_real_batch
                    
                    real_price = y_real_denorm.mean().item()
                    
                    pred_vol_acc += pred_vol
                    pred_price_acc += pred_price
                    real_price_acc += real_price
                    num_samples_epoch += 1

                # Médias
                avg_total = train_loss_acc / len(self.train_loader)
                avg_data = data_loss_acc / len(self.train_loader)
                avg_pde = pde_loss_acc / len(self.train_loader)

                # Validação
                val_mse, val_mae = self.validate()
                
                # Médias das métricas extras
                avg_pred_vol = pred_vol_acc / num_samples_epoch if num_samples_epoch > 0 else 0
                avg_pred_price = pred_price_acc / num_samples_epoch if num_samples_epoch > 0 else 0
                avg_real_price = real_price_acc / num_samples_epoch if num_samples_epoch > 0 else 0
                diff_pred = avg_pred_price - avg_real_price

                # Atualiza histórico
                self.history['train_loss'].append(avg_total)
                self.history['val_loss'].append(val_mse)
                self.history['weight_pde_curr'].append(w_pde)
                self.history['loss_data'].append(avg_data)
                self.history['loss_pde'].append(avg_pde)
                self.history['lr'].append(lr)
                self.history['avg_pred_vol'].append(avg_pred_vol)
                self.history['avg_pred_price'].append(avg_pred_price)
                self.history['avg_real_price'].append(avg_real_price)
                self.history['val_mae'].append(val_mae)
                self.history['bias_pred'].append(diff_pred)
                
                # Razão PDE/Data
                pde_data_ratio = avg_pde / avg_data if avg_data > 0 else 0
                self.history['pde_data_ratio'].append(pde_data_ratio)
                
                # Pesos adaptativos
                if self.use_adaptive_weights:
                    sigma_data = torch.exp(self.log_vars[0]).item()
                    sigma_pde = torch.exp(self.log_vars[1]).item()
                    self.history['sigma_data'].append(sigma_data)
                    self.history['sigma_pde'].append(sigma_pde)
                else:
                    self.history['sigma_data'].append(None)
                    self.history['sigma_pde'].append(None)
                
                # Monitorar bias aprendível
                learned_bias = self.model.price_bias.item()
                self.history['learned_bias'].append(learned_bias)
                
                # Monitorar preservação de moneyness (apenas informativo)
                self.history['moneyness_preservation'].append(self.data_stats.get('moneyness_mean', 0.0))
                
                pbar.set_postfix({
                    'L_Tot': f"{avg_total:.4f}",
                    'L_Dat': f"{avg_data:.4e}",
                    'L_PDE': f"{avg_pde:.4e}",
                    'L_Val': f"{val_mse:.4e}",
                    'MAE': f"{val_mae:.2f}", 
                    'Bias': f"{diff_pred:+.2f}",
                    'Bias_Param': f"{learned_bias:+.4f}"
                })
                
                # Early Stopping
                min_delta = self.config.get('min_delta', 1e-4)
                patience = self.config.get('patience', 15)
                
                if val_mse < best_phase_loss - min_delta:
                    best_phase_loss = val_mse
                    epochs_no_improve = 0
                    
                    if val_mse < self.best_val_loss:
                        self.best_val_loss = val_mse
                        os.makedirs(PATHS.get('model_save_dir', 'resultados'), exist_ok=True)
                        torch.save(self.model.state_dict(), 
                                 os.path.join(PATHS.get('model_save_dir', 'resultados'), 'best_model_weights.pth'))
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logger.warning(f"Estagnação detectada na Fase {phase+1} (Epoch {epoch}). Avançando.")
                    break
            
            pbar.close()
            
        total_time = (time.time() - total_start_time) / 60
        logger.info(f"Treinamento Completo Finalizado em {total_time:.2f} minutos.")
        logger.info(f"Melhor Loss de Validação: {self.best_val_loss:.6f}")
        
        # Salva histórico final
        salvar_historico_treinamento(self.history)

    def validate(self):
        self.model.eval()
        total_mse = 0
        total_mae = 0
        count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x_seq, x_phy, y_norm, _, weights, ids = batch
                x_seq, x_phy, y_norm, ids = x_seq.to(self.device), x_phy.to(self.device), y_norm.to(self.device), ids.to(self.device)
                
                out = self.model(x_seq, x_phy, ids)
                pred_norm = out['price']
                
                # Desnormalização para métricas reais (R$)
                K_norm = x_phy[:, 1:2]
                # CORRECTED: Usar Z-score desnormalização (consistente com data_loader)
                K_real = K_norm * self.data_stats['K_std'] + self.data_stats['K_mean']
                
                pred_price = pred_norm * K_real
                real_price = y_norm * K_real
                
                total_mse += torch.sum((pred_price - real_price)**2).item()
                total_mae += torch.sum(torch.abs(pred_price - real_price)).item()
                count += len(y_norm)
                
        mse = total_mse / count if count > 0 else 0
        mae = total_mae / count if count > 0 else 0
        
        return mse, mae