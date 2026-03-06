# /src/trainer.py

import torch
import torch.nn as nn
import numpy as np
import time
import os
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.physics import heston_residual
from src.config import TRAINING_CONFIG, PATHS
from src.utils import salvar_historico_treinamento, validate_normalization
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
            'moneyness_preservation': [],
            # Parâmetros Heston médios por época (saída da LSTM)
            'avg_nu': [],
            'avg_theta': [],
            'avg_kappa': [],
            'avg_xi': [],
            'avg_rho': [],
        }        
        self.best_val_loss = float('inf')

        # --- Pesos Adaptativos (Learnable Uncertainty) ---
        self.use_adaptive_weights = self.config.get('use_adaptive_weights', False)
        if self.use_adaptive_weights:
            # Parâmetros treináveis: log(sigma^2) para Data e PDE
            self.log_vars = nn.Parameter(torch.zeros(2, device=self.device, requires_grad=True))
    
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
            # Embedding check (opcional, se o modelo usar)
            if hasattr(self.model, 'use_embedding') and self.model.use_embedding:
                 if asset_ids.shape[0] != x_seq.shape[0]:
                    logger.warning(
                        f"Mismatch dimensão asset_ids ({asset_ids.shape[0]}) vs "
                        f"x_seq ({x_seq.shape[0]}). Ajustando..."
                    )
                    asset_ids = asset_ids[:x_seq.shape[0]]
        
        # Habilita gradiente para física
        x_phy.requires_grad_(True)
        
        # Forward Pass
        outputs = self.model(x_seq, x_phy, asset_ids)  
        pred_price = outputs['price']  # Preço normalizado (P/K) ou semi-normalizado
        
        # 1. Recupera os valores reais (Desnormalização) para calcular Loss em Reais
        K_norm = x_phy[:, 1:2]
        # Recupera K_real usando a estatística correta (Std + Mean)
        K_real = K_norm * self.data_stats['K_std'] + self.data_stats['K_mean']
        
        pred_price_real = pred_price * K_real
        target_price_real = y_real * K_real 

        # 2. Loss de Dados (Huber Loss Ponderado)
        # Huber Loss (delta=1.0) é mais robusto a outliers (opções de alto preço) 
        # e força o modelo a focar no erro absoluto quando o erro é grande, 
        # ajudando a derrubar o viés de +1.23.
        huber_fn = nn.HuberLoss(reduction='none', delta=1.0)
        loss_huber_raw = huber_fn(pred_price_real, target_price_real)
        loss_data = torch.mean(weights * loss_huber_raw)

        weight_data = self.config['weight_data']
        
        # 3. Loss da PDE (Heston)
        # Precisamos escalar o output para a PDE ver o preço real também, se necessário
        # A função heston_residual lida com a física internamente, mas passamos o output original
        # Se a PDE esperar preço real, ajustamos aqui. Por padrão, mantemos consistência com model.py
        
        # Nota: Ajustamos o output para passar o preço real para a PDE
        out_phy = outputs.copy()
        out_phy['price'] = pred_price_real 
        
        lambda_bc = self.config.get('lambda_bc', 1.0)
        lambda_reg = self.config.get('lambda_reg', 0.01)
        lambda_feller = self.config.get('lambda_feller', 1.0)
        
        physics_output = heston_residual(
            out_phy, # Passa o output com preço real
            x_phy, 
            self.data_stats,
            lambda_bc=lambda_bc,
            lambda_reg=lambda_reg
        )
        
        loss_pde_total = physics_output['total']
        loss_pde = physics_output['pde']
        loss_bc = physics_output.get('bc', torch.tensor(0.0, device=self.device))
        loss_reg = physics_output.get('reg', torch.tensor(0.0, device=self.device))
        
        # 4. Combinação com Curriculum Learning
        if self.use_adaptive_weights:
            # Pesos adaptativos (Kendall et al.)
            precision_data = torch.exp(-self.log_vars[0])
            precision_pde = torch.exp(-self.log_vars[1])
            
            total_loss = (precision_data * loss_data + self.log_vars[0]) + \
                         (precision_pde * loss_pde_total + self.log_vars[1])
        else:
            # Pesos fixos com curriculum learning
            total_loss = self.config['weight_data'] * loss_data + weight_pde_curr * loss_pde_total

        return total_loss, loss_data, loss_pde_total, loss_bc, loss_reg, outputs

    def train(self):
        logger.info(f"Iniciando Treinamento Híbrido no dispositivo: {self.device}")
        
        # ==========================================================
        # CALLBACK: VALIDAÇÃO DE NORMALIZAÇÃO PRÉ-TREINO
        # ==========================================================
        try:
            validate_normalization(
                train_loader=self.train_loader,
                data_stats=self.data_stats,
                raise_on_critical=True,
            )
        except ValueError as e:
            logger.error(f"Treinamento ABORTADO por falha na validação de normalização:\n{e}")
            raise
        
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
                bc_loss_acc = 0
                reg_loss_acc = 0
                # Acumuladores Heston
                nu_acc = theta_acc = kappa_acc = xi_acc = rho_acc = 0.0
                
                # Acumuladores para métricas de preço/vol
                pred_vol_acc = 0
                pred_price_acc = 0
                real_price_acc = 0
                num_samples_epoch = 0

                # Loop de Batches
                for batch in self.train_loader:
                    self.optimizer.zero_grad()
                    
                    loss, l_data, l_pde, l_bc, l_reg, outputs = self.compute_loss(batch, w_pde)
                    
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
                    bc_loss_acc += l_bc.item()
                    reg_loss_acc += l_reg.item()
                    
                    # Coleta métricas
                    if 'heston_params' in outputs and outputs['heston_params'] is not None:
                        nu, theta, kappa, xi, rho = outputs['heston_params']
                        pred_vol = torch.sqrt(torch.clamp(nu, min=1e-6)).mean().item()
                        # Acumula parâmetros Heston médios do batch
                        nu_acc    += nu.mean().item()
                        theta_acc += theta.mean().item()
                        kappa_acc += kappa.mean().item()
                        xi_acc    += xi.mean().item()
                        rho_acc   += rho.mean().item()
                    else:
                        pred_vol = 0.0
                    
                    # Médias de preço para diagnóstico de viés
                    pred_price_norm = outputs['price']
                    
                    # Desnormaliza para logar
                    x_phy_batch = batch[1].to(self.device)
                    K_norm = x_phy_batch[:, 1:2]
                    K_real = K_norm * self.data_stats['K_std'] + self.data_stats['K_mean']
                    
                    pred_price_real = pred_price_norm * K_real
                    
                    # Target Real
                    y_real_batch = batch[2].to(self.device)
                    target_price_real = y_real_batch * K_real
                    
                    pred_price_mean = pred_price_real.mean().item()
                    real_price_mean = target_price_real.mean().item()
                    
                    pred_vol_acc += pred_vol
                    pred_price_acc += pred_price_mean
                    real_price_acc += real_price_mean
                    num_samples_epoch += 1

                # Médias
                avg_total = train_loss_acc / len(self.train_loader) if len(self.train_loader) > 0 else 0
                avg_data = data_loss_acc / len(self.train_loader) if len(self.train_loader) > 0 else 0
                avg_pde = pde_loss_acc / len(self.train_loader) if len(self.train_loader) > 0 else 0
                avg_bc = bc_loss_acc / len(self.train_loader) if len(self.train_loader) > 0 else 0
                avg_reg = reg_loss_acc / len(self.train_loader) if len(self.train_loader) > 0 else 0

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
                self.history['loss_bc'].append(avg_bc)
                self.history['loss_reg'].append(avg_reg)
                self.history['lr'].append(lr)
                self.history['avg_pred_vol'].append(avg_pred_vol)
                self.history['avg_pred_price'].append(avg_pred_price)
                self.history['avg_real_price'].append(avg_real_price)
                self.history['val_mae'].append(val_mae)
                self.history['bias_pred'].append(diff_pred)
                
                # Parâmetros Heston médios da época
                n_batches = len(self.train_loader) if len(self.train_loader) > 0 else 1
                self.history['avg_nu'].append(nu_acc / n_batches)
                self.history['avg_theta'].append(theta_acc / n_batches)
                self.history['avg_kappa'].append(kappa_acc / n_batches)
                self.history['avg_xi'].append(xi_acc / n_batches)
                self.history['avg_rho'].append(rho_acc / n_batches)
                
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
                
                # Monitorar preservação de moneyness (apenas informativo)
                self.history['moneyness_preservation'].append(self.data_stats.get('moneyness_mean', 0.0))
                
                pbar.set_postfix({
                    'Loss': f"{avg_total:.4f}",
                    'Data': f"{avg_data:.4f}",
                    'PDE': f"{avg_pde:.2f}",
                    'MAE': f"{val_mae:.2f}", 
                    'Bias': f"{diff_pred:+.2f}"
                })
                
                # Early Stopping
                min_delta = self.config.get('min_delta', 1e-4)
                patience = self.config.get('patience', 15)
                
                if val_mse < best_phase_loss - min_delta:
                    best_phase_loss = val_mse
                    epochs_no_improve = 0
                    
                    if val_mse < self.best_val_loss:
                        self.best_val_loss = val_mse
                        save_dir = PATHS.get('model_save_dir', 'resultados')
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, 'best_model_weights.pth')
                        
                        # Salvamento robusto para Windows (evita ERROR_USER_MAPPED_FILE 1224)
                        try:
                            tmp_path = save_path + ".tmp"
                            torch.save(self.model.state_dict(), tmp_path)
                            # os.replace é mais atômico e lida melhor com substituição no Windows
                            if os.path.exists(tmp_path):
                                if os.path.exists(save_path):
                                    try: os.remove(save_path)
                                    except: pass
                                os.replace(tmp_path, save_path)
                                logger.info(f"💾 Melhor modelo salvo: {val_mse:.6f}")
                        except Exception as e:
                            logger.warning(f"Erro não fatal ao salvar pesos (lock do Windows?): {e}")
                            # Tenta salvar com nome alternativo se o principal estiver travado
                            try:
                                recovery_path = save_path.replace('.pth', f'_bkp_epoch_{global_epoch}.pth')
                                torch.save(self.model.state_dict(), recovery_path)
                                logger.info(f"💾 Modelo recuperado em: {recovery_path}")
                            except:
                                logger.error("Falha crítica ao salvar pesos. O treino continuará, mas o modelo não foi persistido.")
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
                K_real = K_norm * self.data_stats['K_std'] + self.data_stats['K_mean']
                
                pred_price = pred_norm * K_real
                real_price = y_norm * K_real
                
                # MSE e MAE em escala real
                total_mse += torch.sum((pred_price - real_price)**2).item()
                total_mae += torch.sum(torch.abs(pred_price - real_price)).item()
                count += len(y_norm)
                
        mse = total_mse / count if count > 0 else 0
        mae = total_mae / count if count > 0 else 0
        
        return mse, mae