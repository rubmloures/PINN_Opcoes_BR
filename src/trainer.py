# /src/trainer.py

import torch
import numpy as np
import time
import os
from torch.utils.data import DataLoader

from src.model import PINN_BlackScholes
from src.physics import black_scholes_residual, payoff_boundary_condition
from src.config import TRAINING_CONFIG, PATHS
from src.utils import salvar_historico_treinamento

class PINNTrainer:
    def __init__(self, model: PINN_BlackScholes, train_loader: DataLoader, val_loader: DataLoader, data_stats: dict, config: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_stats = data_stats
        self.config = config
        self.device = torch.device(self.config['device'])
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        self.history = {'train_loss': [], 'val_loss': [], 'lr': [], 'weight_data': [], 'weight_pde': []}
        self.best_val_loss = float('inf')
        self.collocation_points_cache = None

        self.use_adaptive_weights = self.config.get('use_adaptive_weights', False)
        if self.use_adaptive_weights:
            print("Usando Ponderação Adaptativa de Perdas.")
            self.log_lambda_data = torch.nn.Parameter(torch.zeros(1, device=self.device, requires_grad=True))
            self.log_lambda_pde = torch.nn.Parameter(torch.zeros(1, device=self.device, requires_grad=True))
            self.optimizer.add_param_group({'params': [self.log_lambda_data, self.log_lambda_pde]})

    def _resample_collocation_points(self, n_points: int, n_resample: int = 10000):
        """
        Avalia o resíduo da EDP em uma grande grade de pontos e amostra
        os pontos com maior erro para usar no treinamento.
        """
        print("    --> Reamostrando pontos de colocação por importância...")
        self.model.eval() # Coloca o modelo em modo de avaliação para esta etapa
        
        # Gera uma grande quantidade de pontos candidatos
        candidate_points = torch.rand(n_resample, self.model.config['input_size'], device=self.device)
        candidate_points.requires_grad = True
        output_phy = self.model(candidate_points)
        
        # Calcula o resíduo (precisa de gradientes)
        pde_res = black_scholes_residual(output_phy, candidate_points, self.data_stats)
        errors = torch.abs(pde_res).flatten().cpu().detach().numpy()
        
        # Normaliza os erros para que se tornem probabilidades
        probabilities = errors / errors.sum()
        
        # Amostra os índices dos pontos com base nas probabilidades (erros)
        chosen_indices = np.random.choice(n_resample, n_points, p=probabilities)
        
        # Seleciona os pontos de alta importância
        self.collocation_points_cache = candidate_points[chosen_indices].detach()
        self.collocation_points_cache.requires_grad = True
        
        self.model.train() # Retorna o modelo ao modo de treinamento

    def _get_collocation_points(self, n_points: int) -> torch.Tensor:
        if self.collocation_points_cache is None:
            # Na primeira vez, usa amostragem aleatória uniforme
            points = torch.rand(n_points, self.model.config['input_size'], device=self.device)
            points.requires_grad = True
            return points
        else:
            # Nas vezes seguintes, usa o cache de pontos importantes
            return self.collocation_points_cache

    def _compute_train_losses(self, batch_data: tuple, pde_weight: float, data_weight: float) -> dict:
        """Calcula a perda HÍBRIDA apenas para o passo de TREINAMENTO."""
        inputs, premiums_real = batch_data
        inputs, premiums_real = inputs.to(self.device), premiums_real.to(self.device)
        model_output = self.model(inputs)
        price_pred = model_output['price']
        loss_data = torch.nn.functional.mse_loss(price_pred, premiums_real)
        
        n_boundary = inputs.size(0) // 4
        s_boundary = torch.rand(n_boundary, 1, device=self.device)
        k_boundary = torch.rand(n_boundary, 1, device=self.device)
        t_boundary = torch.zeros(n_boundary, 1, device=self.device)
        r_boundary = torch.rand(n_boundary, 1, device=self.device)
        p_boundary = torch.rand(n_boundary, 1, device=self.device)
        boundary_inputs = torch.cat([s_boundary, k_boundary, t_boundary, r_boundary, p_boundary], dim=1)
        
        S_b = s_boundary * (self.data_stats['S_max'] - self.data_stats['S_min']) + self.data_stats['S_min']
        K_b = k_boundary * (self.data_stats['K_max'] - self.data_stats['K_min']) + self.data_stats['K_min']
        
        payoff_real = payoff_boundary_condition(S_b, K_b)
        price_pred_boundary = self.model(boundary_inputs)['price']
        loss_boundary = torch.nn.functional.mse_loss(price_pred_boundary, payoff_real)
        
        collocation_points = self._get_collocation_points(self.config['phy_batch_size'])
        output_phy = self.model(collocation_points)
        pde_res = black_scholes_residual(output_phy, collocation_points, self.data_stats)
        loss_pde = torch.nn.functional.mse_loss(pde_res, torch.zeros_like(pde_res))

        if self.use_adaptive_weights:
            # Ponderação adaptativa (auto-balanceamento via gradiente)
            # A perda é formulada para minimizar as perdas individuais e regularizar os pesos.
            total_loss = (torch.exp(-self.log_lambda_data) * loss_data + self.log_lambda_data) + \
                         (torch.exp(-self.log_lambda_pde) * loss_pde + self.log_lambda_pde) + \
                         loss_boundary # Contorno pode ter peso fixo 1
            
            return {'total': total_loss, 'data_weight': torch.exp(-self.log_lambda_data).item(), 'pde_weight': torch.exp(-self.log_lambda_pde).item()}
        else:
            # Curriculum Learning Manual
            progress = len(self.history['train_loss']) / (len(self.config['learning_rates']) * self.config['max_epochs_per_phase'])
            pde_weight = self.config['initial_pde_weight'] + progress * (self.config['final_pde_weight'] - self.config['initial_pde_weight'])
            data_weight = self.config['initial_data_weight'] - progress * (self.config['initial_data_weight'] - self.config['final_data_weight'])
            total_loss = (loss_data * data_weight) + (loss_pde * pde_weight) + loss_boundary
            return {'total': total_loss, 'data_weight': data_weight, 'pde_weight': pde_weight}


    def train(self):
        print(f"Iniciando treinamento no dispositivo: {self.device}")
        total_start_time = time.time()
        
        learning_rates = self.config.get('learning_rates', [1e-4, 1e-5, 1e-6])
        epochs_per_phase = self.config.get('max_epochs_per_phase', 5000)
        resample_every = self.config.get('resample_every', 25) # Frequência da amostragem por importância
        total_epochs_done = 0
        training_completed = False

        for i, lr in enumerate(learning_rates):
            print(f"\n--- Iniciando Fase de Treinamento {i+1}/{len(learning_rates)} com LR = {lr:.1e} ---")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            epochs_no_improve = 0
            
            for epoch_in_phase in range(epochs_per_phase):
                
                # --- Lógica de Amostragem por Importância ---
                if (total_epochs_done % resample_every) == 0:
                    self._resample_collocation_points(self.config['phy_batch_size'])

                epoch_start_time = time.time()
                self.model.train()

                progress = total_epochs_done / (len(learning_rates) * epochs_per_phase) if (len(learning_rates) * epochs_per_phase) > 0 else 0
                pde_weight = self.config['initial_pde_weight'] + progress * (self.config['final_pde_weight'] - self.config['initial_pde_weight'])
                data_weight = self.config['initial_data_weight'] - progress * (self.config['initial_data_weight'] - self.config['final_data_weight'])

                total_epoch_loss = 0
                for batch_data in self.train_loader:
                    self.optimizer.zero_grad()
                    losses = self._compute_train_losses(batch_data, pde_weight, data_weight)
                    losses['total'].backward()
                    self.optimizer.step()
                    total_epoch_loss += losses['total'].item()
                
                avg_train_loss = total_epoch_loss / len(self.train_loader)

                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_data in self.val_loader:
                        inputs, premiums_real = batch_data
                        inputs, premiums_real = inputs.to(self.device), premiums_real.to(self.device)
                        price_pred = self.model(inputs)['price']
                        val_loss = torch.nn.functional.mse_loss(price_pred, premiums_real)
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(self.val_loader)

                self.history['train_loss'].append(avg_train_loss)
                self.history['val_loss'].append(avg_val_loss)
                self.history['lr'].append(lr)
                self.history['weight_data'].append(losses.get('data_weight', 0))
                self.history['weight_pde'].append(losses.get('pde_weight', 0))

                total_epochs_done += 1
                
                print(f"Época Total: {total_epochs_done:04d} | Fase {i+1} | "
                      f"Tempo: {time.time() - epoch_start_time:.2f}s | LR: {lr:.1e} | "
                      f"Loss Treino: {avg_train_loss:.4e} | Loss Val: {avg_val_loss:.4e}")

                if avg_val_loss < self.best_val_loss - self.config['min_delta']:
                    self.best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    os.makedirs(PATHS['model_save_dir'], exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(PATHS['model_save_dir'], 'best_model_weights.pth'))
                    print(f"    ==> Nova melhor Loss de Validação Global: {self.best_val_loss:.4e}. Modelo salvo!")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.config['patience']:
                    is_last_phase = (i == len(learning_rates) - 1)
                    if is_last_phase:
                        print("\nParada antecipada definitiva na última fase de LR.")
                        training_completed = True
                    else:
                        print(f"Paciência atingida na fase {i+1}. Avançando para a próxima fase de LR.")
                    break
            
            if training_completed:
                break
        
        if not training_completed:
            print("\nTreinamento concluído após todas as fases de LR.")
        
        total_time = (time.time() - total_start_time) / 60
        print(f"\nTreinamento finalizado em {total_time:.2f} minutos.")
        salvar_historico_treinamento(self.history)