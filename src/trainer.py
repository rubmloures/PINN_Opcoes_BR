# /src/trainer.py

import torch
import numpy as np
import time
import os
from tqdm import tqdm
import time

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

        self.history = {'train_loss': [], 'val_loss': [], 'lr': [], 'weight_data': [], 'weight_pde': [], 'weight_reg': []}        
        self.best_val_loss = float('inf')
        self.collocation_points_cache = None

        self.use_adaptive_weights = self.config.get('use_adaptive_weights', False)
        if self.use_adaptive_weights:
            self.log_lambda_data = torch.nn.Parameter(torch.zeros(1, device=self.device, requires_grad=True))
            self.log_lambda_pde = torch.nn.Parameter(torch.zeros(1, device=self.device, requires_grad=True))
            self.log_lambda_reg = torch.nn.Parameter(torch.zeros(1, device=self.device, requires_grad=True))
            self.optimizer.add_param_group({'params': [self.log_lambda_data, self.log_lambda_pde, self.log_lambda_reg]})

    def _resample_collocation_points(self, n_points: int, n_resample: int = 10000):
        """
        Gera muitos pontos candidatos no espaço (S_norm,K_norm,T_norm,r), seleciona por importância
        com base no resíduo PDE e mantém um cache dos melhores pontos.
        Observação: aqui NÃO usamos valores reais de premium — preenchimento com zero (dummy)
        para compatibilidade com a interface do modelo (5 colunas).
        """
        #print("    --> Reamostrando pontos de colocação por importância...")
        self.model.eval()

        # input_price_size = 4 (S_norm, K_norm, T_norm, r)
        input_price_size = self.model.config['input_size'] - 1

        # Gera candidatos no espaço de 4 variáveis (normalizado)
        candidate_price = torch.rand(n_resample, input_price_size, device=self.device)
        candidate_price.requires_grad = True

        # Anexa coluna dummy de premium (zeros) para formar vetores de 5 colunas compatíveis
        zeros_col = torch.zeros(n_resample, 1, device=self.device)
        candidate_points = torch.cat([candidate_price, zeros_col], dim=1)
        candidate_points = candidate_points.detach().clone().requires_grad_(True)

        # Avalia resíduo PDE nesses candidatos
        with torch.no_grad():
            # Permite que o modelo calcule outputs; para cálculo do PDE sem grad interno aqui usamos o grafo depois
            pass

        # Precisamos de grad para o cálculo do resíduo (black_scholes_residual faz autograd)
        candidate_points.requires_grad = True
        output_phy = self.model(candidate_points)
        pde_res = black_scholes_residual(output_phy, candidate_points, self.data_stats)
        errors = torch.abs(pde_res).flatten().cpu().detach().numpy()

        # Normaliza as probabilidades
        if errors.sum() == 0:
            probabilities = np.ones_like(errors) / errors.size
        else:
            probabilities = errors / errors.sum()

        # Amostra índices com base nas probabilidades
        chosen_indices = np.random.choice(n_resample, n_points, p=probabilities)

        # Armazena cache (desanexado do grafo, mas com requires_grad True para próximo uso)
        self.collocation_points_cache = candidate_points[chosen_indices].detach()
        self.collocation_points_cache.requires_grad = True

        self.model.train()

    def _get_collocation_points(self, n_points: int) -> torch.Tensor:
        """
        Retorna pontos de colocação (com 5 colunas, última coluna premium dummy=0).
        Se não houver cache, gera amostra uniforme no espaço de 4 variáveis e anexa a coluna dummy.
        """
        if self.collocation_points_cache is None:
            input_price_size = self.model.config['input_size'] - 1
            points_price = torch.rand(n_points, input_price_size, device=self.device)
            zeros_col = torch.zeros(n_points, 1, device=self.device)
            points = torch.cat([points_price, zeros_col], dim=1)
            points.requires_grad = True
            return points
        else:
            return self.collocation_points_cache

    def _compute_train_losses(self, batch_data: tuple, pde_weight: float, data_weight: float) -> dict:
        """
        Calcula perdas:
        - loss_data: MSE entre price_pred e premium_real (supervisionado)
        - loss_boundary: condição de payoff no vencimento
        - loss_pde: MSE do resíduo PDE em collocation points (entrada sem premium real)
        - loss_reg_sigma: Perda de suavidade da superfície sigma
        """
        inputs, premiums_real = batch_data
        inputs, premiums_real = inputs.to(self.device), premiums_real.to(self.device)

        # Forward no modelo -> outputs contém 'price' e 'sigma'
        model_output = self.model(inputs)
        price_pred = model_output['price']
        loss_data = torch.nn.functional.mse_loss(price_pred, premiums_real)

        # --- Boundary loss: cria inputs de boundary com premium dummy zero ---
        n_boundary = max(1, inputs.size(0) // 4)
        s_boundary = torch.rand(n_boundary, 1, device=self.device)
        k_boundary = torch.rand(n_boundary, 1, device=self.device)
        t_boundary = torch.zeros(n_boundary, 1, device=self.device)  # t=0 (vencimento)
        r_boundary = torch.rand(n_boundary, 1, device=self.device)
        p_boundary = torch.zeros(n_boundary, 1, device=self.device)  # premium dummy (não mercado)
        boundary_inputs = torch.cat([s_boundary, k_boundary, t_boundary, r_boundary, p_boundary], dim=1)

        # Desnormaliza para payoff real
        S_b = s_boundary * (self.data_stats['S_max'] - self.data_stats['S_min']) + self.data_stats['S_min']
        K_b = k_boundary * (self.data_stats['K_max'] - self.data_stats['K_min']) + self.data_stats['K_min']
        payoff_real = payoff_boundary_condition(S_b, K_b)

        price_pred_boundary = self.model(boundary_inputs)['price']
        loss_boundary = torch.nn.functional.mse_loss(price_pred_boundary, payoff_real)

        # --- PDE loss e REG loss (nos pontos de colocação) ---
        collocation_points = self._get_collocation_points(self.config['phy_batch_size'])
        output_phy = self.model(collocation_points)
        
        # Cálculo da Regularização de Suavidade do Sigma ---
        sigma_phy = output_phy['sigma']
        
        # Calcula gradientes do sigma em relação às entradas (S, K, T, r, dummy)
        # retain_graph=True é CRÍTICO, pois o cálculo do 'pde_res' logo abaixo
        # também precisará usar o grafo computacional.
        grad_sigma = torch.autograd.grad(
            sigma_phy, collocation_points,
            grad_outputs=torch.ones_like(sigma_phy),
            create_graph=True,
            retain_graph=True 
        )[0]
        
        # Pegamos as derivadas em relação a S (idx 0), K (idx 1), e T (idx 2)
        grad_sigma_S = grad_sigma[:, 0]
        grad_sigma_K = grad_sigma[:, 1]
        grad_sigma_T = grad_sigma[:, 2]
        
        # A perda é o MSE dessas derivadas (forçando-as a ser perto de zero, ou seja, "suave")
        loss_reg_sigma = torch.mean(torch.square(grad_sigma_S)) + \
                         torch.mean(torch.square(grad_sigma_K)) + \
                         torch.mean(torch.square(grad_sigma_T))
        # Cálculo da PDE Loss
        pde_res = black_scholes_residual(output_phy, collocation_points, self.data_stats)
        loss_pde = torch.nn.functional.mse_loss(pde_res, torch.zeros_like(pde_res))
    
        # --- Combina perdas ---
        if self.use_adaptive_weights:
            loss_term_reg = (torch.exp(-self.log_lambda_reg) * loss_reg_sigma + self.log_lambda_reg)
            total_loss = (
                (torch.exp(-self.log_lambda_data) * loss_data + self.log_lambda_data) +
                (torch.exp(-self.log_lambda_pde) * loss_pde + self.log_lambda_pde) +
                loss_term_reg + 
                loss_boundary
            )
            return {
                'total': total_loss,
                'data': loss_data.item(),
                'pde': loss_pde.item(),
                'reg': loss_reg_sigma.item(), 
                'data_weight': torch.exp(-self.log_lambda_data).item(),
                'pde_weight': torch.exp(-self.log_lambda_pde).item(),
                'reg_weight': torch.exp(-self.log_lambda_reg).item() 
            }
    
        else:
            reg_weight = self.config.get('sigma_reg_weight', 0.01) 
            total_loss = (
                (loss_data * data_weight) + 
                (loss_pde * pde_weight) + 
                (loss_reg_sigma * reg_weight) + 
                loss_boundary
            )
            return {
                'total': total_loss,
                'data': loss_data.item(),
                'pde': loss_pde.item(),
                'reg': loss_reg_sigma.item(), 
                'data_weight': data_weight,
                'pde_weight': pde_weight,
                'reg_weight': reg_weight 
            }

    def train(self):
        print(f"Iniciando treinamento no dispositivo: {self.device}")
        total_start_time = time.time()
        # Hiperparâmetros principais
        learning_rates = self.config.get('learning_rates', [1e-4, 1e-5, 1e-6])
        epochs_per_phase = self.config.get('epochs_per_phase', self.config.get('max_epochs_per_phase', 5000))
        resample_every = self.config.get('resample_every', 25)
        total_epochs_done = 0
        training_completed = False
        # Loop sobre diferentes fases de taxa de aprendizado
        for i, lr in enumerate(learning_rates):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            epochs_no_improve = 0
            # Criação da barra de progresso para a fase atual
            pbar = tqdm(range(epochs_per_phase),
                    desc=f"Fase {i+1}/{len(learning_rates)} (LR: {lr:.1e})",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [Tempo: {elapsed}<{remaining}] {postfix}'
                )
            
            for epoch_in_phase in range(epochs_per_phase):
                # Reamostragem por importância periódica
                if (total_epochs_done % resample_every) == 0:
                    self._resample_collocation_points(self.config['phy_batch_size'])

                epoch_start_time = time.time()
                self.model.train()

                # pesos para curriculum (pde vs data) calculados externamente no loop
                progress = total_epochs_done / (len(learning_rates) * epochs_per_phase) if (len(learning_rates) * epochs_per_phase) > 0 else 0
                pde_weight = self.config['initial_pde_weight'] + progress * (self.config['final_pde_weight'] - self.config['initial_pde_weight'])
                data_weight = self.config['initial_data_weight'] - progress * (self.config['initial_data_weight'] - self.config['final_data_weight'])

                total_loss_epoch = 0.0
                total_loss_data = 0.0
                total_loss_pde = 0.0
                total_loss_reg = 0.0
                n_batches = len(self.train_loader)

                for batch_data in self.train_loader:
                    self.optimizer.zero_grad()
                    # Calcula perdas (agora retorna um dict com valores .item())
                    losses = self._compute_train_losses(batch_data, pde_weight, data_weight)
                    # Backpropagation
                    losses['total'].backward()
                    self.optimizer.step()
                    # Acumula perdas para a média da época
                    total_loss_epoch += losses['total'].item()
                    total_loss_data += losses['data']
                    total_loss_pde += losses['pde']
                    total_loss_reg += losses['reg']

                avg_loss_total = total_loss_epoch / n_batches
                avg_loss_data = total_loss_data / n_batches
                avg_loss_pde = total_loss_pde / n_batches
                avg_loss_reg = total_loss_reg / n_batches

                # Validação curta (apenas data loss)
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch_data in self.val_loader:
                        inputs, premiums_real = batch_data
                        inputs, premiums_real = inputs.to(self.device), premiums_real.to(self.device)
                        price_pred = self.model(inputs)['price']
                        val_loss = torch.nn.functional.mse_loss(price_pred, premiums_real)
                        total_val_loss += val_loss.item()
                avg_val_loss = total_val_loss / max(1, len(self.val_loader))

                 # Registro de histórico
                self.history['train_loss'].append(avg_loss_total)
                self.history['val_loss'].append(avg_val_loss)
                self.history['lr'].append(lr)
                
                # Obtém pesos da fase (seja adaptativo ou curriculum)
                self.history['weight_data'].append(losses['data_weight'])
                self.history['weight_pde'].append(losses['pde_weight'])
                self.history['weight_reg'].append(losses['reg_weight'])

                total_epochs_done += 1

                # Atualiza barra de progresso com informações essenciais
                postfix_str = (
                    f"L_Data: {avg_loss_data:.2e}, "
                    f"L_PDE: {avg_loss_pde:.2e}, "
                    f"L_Reg: {avg_loss_reg:.2e} | "  # Separador
                    f"L_Val: {avg_val_loss:.2e} (Best: {self.best_val_loss:.2e})"
                )
                pbar.set_postfix_str(postfix_str)
                pbar.update(1)
                # Early stopping / salvar melhor modelo
                if avg_val_loss < self.best_val_loss - self.config['min_delta']:
                    self.best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    os.makedirs(PATHS['model_save_dir'], exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(PATHS['model_save_dir'], 'best_model_weights.pth'))
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.config['patience']:
                    is_last_phase = (i == len(learning_rates) - 1)
                    if is_last_phase:
                        training_completed = True
                    break
            
            pbar.close()  # Fecha barra da fase
            if training_completed:
                break

        if not training_completed:
            # finalização normal após todas as fases
            pass

        total_time = (time.time() - total_start_time) / 60
        print(f"\nTreinamento finalizado em {total_time:.2f} minutos.")
        salvar_historico_treinamento(self.history)
