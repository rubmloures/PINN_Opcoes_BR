# /src/fine_tuner.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from src.logger import get_logger
from src.config import TRAINING_CONFIG

logger = get_logger('PINN_FineTuner')


class FineTuner:
    """
    Fine-tunes o modelo robusto para especialização por ativo.
    
    Congela a rede PINN (módulos C e B) e libera a LSTM (módulo A)
    para adaptar a calibração de volatilidade e correlação a cada ativo.
    """

    def __init__(self, base_model, full_dataset, data_stats, config, paths):
        """
        Inicializa o Fine-Tuner.
        
        Args:
            base_model: Modelo robusto já treinado (DeepHestonHybrid)
            full_dataset: Dataset completo (TensorDataset com 6 elementos)
            data_stats: Dict com estatísticas e 'asset_map'
            config: Dicionário de configurações de treino
            paths: Dicionário com caminhos de diretórios
            
        Raises:
            ValueError: Se asset_map ausente ou dataset inválido
        """
        self.base_model = base_model
        self.full_dataset = full_dataset
        self.data_stats = data_stats
        self.config = config
        self.paths = paths
        self.device = next(base_model.parameters()).device
        
        # --- Validar asset_map ---
        if 'asset_map' not in data_stats:
            raise ValueError(
                "data_stats não contém 'asset_map'. "
                "Verifique se criar_dataset_hibrido() foi usado corretamente."
            )
        
        self.asset_map = data_stats['asset_map']
        if not self.asset_map:
            raise ValueError("asset_map está vazio!")
        
        logger.info(f"Asset map carregado: {list(self.asset_map.keys())}")
        
        # --- Validar estrutura do dataset ---
        if len(full_dataset[0]) != 6:
            raise ValueError(
                f"Dataset deve ter 6 elementos, encontrado {len(full_dataset[0])}. "
                f"Estrutura esperada: X_seq, X_phy, y, X_time, weights, X_asset"
            )
        
        # Índice do asset_id no TensorDataset (conforme data_loader.py)
        # Ordem: X_seq, X_phy, y, X_time, weights, X_asset
        self.asset_col_idx = 5
        
        logger.info(f"FineTuner inicializado. Device: {self.device}")

    def _freeze_pinn_layers(self):
        """
        Congela a rede PINN (pricing_net + fourier_layer).
        Libera LSTM + Asset Embeddings para fine-tuning por ativo.
        
        Referência: Documentação Técnica, Seção 4 (Fine-Tuning)
        
        Raises:
            RuntimeError: Se validação de freezing falhar
            
        Returns:
            bool: True se freezing bem-sucedido
        """
        logger.info("Congelando pesos da PINN (pricing_net)...")
        for name, param in self.base_model.pricing_net.named_parameters():
            param.requires_grad = False
        
        # Congelar Fourier Layer (Módulo B)
        if hasattr(self.base_model, 'fourier_layer') and self.base_model.fourier_layer is not None:
            logger.info("Congelando pesos da Fourier Layer...")
            for name, param in self.base_model.fourier_layer.named_parameters():
                param.requires_grad = False
        
        # Liberar LSTM e Heston Parameter Head (Módulo A)
        logger.info("Liberando pesos da LSTM e Heston Parameter Head...")
        for name, param in self.base_model.lstm.named_parameters():
            param.requires_grad = True
        for name, param in self.base_model.heston_head.named_parameters():
            param.requires_grad = True
        
        # Liberar Asset Embeddings
        if hasattr(self.base_model, 'asset_embedding') and self.base_model.asset_embedding is not None:
            logger.info("Liberando pesos do Asset Embedding Layer...")
            for name, param in self.base_model.asset_embedding.named_parameters():
                param.requires_grad = True
        
        # --- VALIDAÇÃO STRICTA ---
        trainable_count = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        frozen_count = sum(p.numel() for p in self.base_model.parameters() if not p.requires_grad)
        
        if trainable_count == 0:
            raise RuntimeError("Erro: Nenhum parâmetro está treinável após freezing!")
        if frozen_count == 0:
            raise RuntimeError("Erro: Nenhum parâmetro está congelado após freezing!")
        
        # Validação específica: PINN congelada, LSTM treinável
        pinn_frozen = all(not p.requires_grad for p in self.base_model.pricing_net.parameters())
        lstm_trainable = any(p.requires_grad for p in self.base_model.lstm.parameters())
        
        if not pinn_frozen:
            raise RuntimeError("Erro: PINN não foi completamente congelada!")
        if not lstm_trainable:
            raise RuntimeError("Erro: LSTM não tem parâmetros treináveis!")
        
        logger.info(f"✓ Parâmetros após freezing - Treináveis: {trainable_count:,}, Congelados: {frozen_count:,}")
        logger.info(f"✓ PINN congelada: {pinn_frozen}")
        logger.info(f"✓ LSTM treinável: {lstm_trainable}")
        
        return trainable_count > 0 and frozen_count > 0

    def fine_tune_all(self, epochs: int = None, lr: float = None) -> dict:
        """
        Fine-tune o modelo por ativo, congelando a PINN e liberando LSTM/Embeddings.
        
        Fluxo por Ativo:
        1. Congela pricing_net e fourier_layer (PINN - Módulos B e C)
        2. Libera lstm, heston_head, asset_embedding (LSTM - Módulo A)
        3. Filtra dados específicos do ativo
        4. Treina por N épocas com early stopping
        5. Carrega melhor checkpoint
        
        Args:
            epochs (int): Número de épocas por ativo. 
                         Se None, lê de TRAINING_CONFIG['finetune_epochs']
            lr (float): Learning rate para fine-tuning.
                       Se None, lê de TRAINING_CONFIG['finetune_learning_rate']
        
        Returns:
            dict: Histórico de loss por ativo
                 {
                     'PETR4': {'loss': [...], 'mae': [...]},
                     'VALE3': {'loss': [...], 'mae': [...]},
                     ...
                 }
        
        Raises:
            ValueError: Se asset_map vazio ou dataset inválido
            RuntimeError: Se freezing falhar
            
        Referência: Documentação Técnica, Seção 4 (Fine-Tuning)
        """
        # --- P1.1: Ler defaults de TRAINING_CONFIG se não fornecidos ---
        if epochs is None:
            epochs = TRAINING_CONFIG.get('finetune_epochs', 10)
        if lr is None:
            lr = TRAINING_CONFIG.get('finetune_learning_rate', 1e-4)
        
        # --- P2.3: Ler min_samples de config ---
        min_samples = self.config.get('finetune_min_samples', 
                                      TRAINING_CONFIG.get('finetune_min_samples', 50))
        
        logger.info(f"Fine-tuning com: epochs={epochs}, lr={lr}, min_samples={min_samples}")
        
        self.base_model = self.base_model.to(self.device)
        all_history = {}
        
        logger.info("\n" + "="*70)
        logger.info("MODO AUTOMÁTICO: Fine-Tuning para TODOS os ativos")
        logger.info("="*70)
        
        for asset_name in self.asset_map.keys():
            logger.info(f"\n{'='*70}")
            logger.info(f"Iniciando Fine-tuning: {asset_name}")
            logger.info(f"{'='*70}")
            
            asset_id = self.asset_map[asset_name]
            
            # ===== CONGELAMENTO DE PESOS =====
            try:
                if not self._freeze_pinn_layers():
                    raise RuntimeError(f"Falha na validação de freezing para {asset_name}")
            except RuntimeError as e:
                logger.error(f"Erro ao congelar camadas para {asset_name}: {e}")
                raise
            
            # Coletar parâmetros treináveis
            trainable_params = [p for p in self.base_model.parameters() if p.requires_grad]
            
            # ===== FILTRAR DADOS DO ATIVO =====
            all_ids = self.full_dataset.tensors[self.asset_col_idx].numpy()
            indices = np.where(all_ids == asset_id)[0]
            
            if len(indices) < min_samples:
                logger.warning(
                    f"Poucos dados para {asset_name} ({len(indices)} < {min_samples}). "
                    f"Pulando fine-tuning."
                )
                continue
            
            # Split 80/20
            split = int(len(indices) * 0.8)
            train_idx, val_idx = indices[:split], indices[split:]
            
            # --- P1.2: Usar finetune_batch_size em vez de batch_size global ---
            batch_size = self.config.get('finetune_batch_size', 
                                        TRAINING_CONFIG.get('finetune_batch_size', 256))
            
            asset_loader = DataLoader(
                Subset(self.full_dataset, train_idx), 
                batch_size=batch_size,
                shuffle=True
            )
            
            logger.info(
                f"Dados do ativo: {len(train_idx)} treino, {len(val_idx)} validação, "
                f"batch_size={batch_size}"
            )
            
            # ===== OTIMIZADOR E SCHEDULER =====
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=1e-4,
                betas=(0.9, 0.999)
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=lr * 0.1
            )
            
            # ===== TREINO POR ATIVO =====
            phase_history = {'loss': [], 'mae': []}
            best_loss = float('inf')
            patience_counter = 0
            
            # --- P1.3: Usar finetune_patience de config ---
            patience = self.config.get('finetune_patience', 
                                      TRAINING_CONFIG.get('finetune_patience', 10))
            
            logger.info(f"Early stopping patience={patience}")
            
            for epoch in range(epochs):
                self.base_model.train()
                epoch_loss = 0.0
                epoch_mae = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(asset_loader):
                    optimizer.zero_grad()
                    
                    # Desempacotar batch (6-tuple conforme data_loader.py)
                    x_seq, x_phy, y_real, X_time, weights, asset_ids_batch = batch
                    
                    # Mover para device
                    x_seq = x_seq.to(self.device)
                    x_phy = x_phy.to(self.device)
                    y_real = y_real.to(self.device)
                    weights = weights.to(self.device)
                    asset_ids_batch = asset_ids_batch.to(self.device) if asset_ids_batch is not None else None
                    
                    # Forward pass
                    outputs = self.base_model(x_seq, x_phy, asset_ids_batch)
                    y_pred = outputs['price']
                    
                    # Loss MSE ponderado por moneyness
                    loss_mse = torch.mean(weights * (y_pred - y_real) ** 2)
                    
                    # Backward
                    loss_mse.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Métricas
                    with torch.no_grad():
                        mae = torch.mean(torch.abs(y_pred - y_real))
                        epoch_loss += loss_mse.item()
                        epoch_mae += mae.item()
                        num_batches += 1
                
                # Média da época
                avg_loss = epoch_loss / num_batches
                avg_mae = epoch_mae / num_batches
                
                phase_history['loss'].append(avg_loss)
                phase_history['mae'].append(avg_mae)
                
                scheduler.step()
                
                # --- P3.1: Melhorar formatação do logging ---
                logger.info(
                    f"{asset_name:12s} | Epoch {epoch+1:3d}/{epochs:3d} | "
                    f"Loss: {avg_loss:10.6f} | MAE: {avg_mae:10.6f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    
                    # Salvar melhor modelo
                    save_dir = self.paths.get('model_save_dir', 
                                             self.config.get('model_save_dir', 'resultados'))
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = f"{save_dir}/best_{asset_name}.pt"
                    torch.save(self.base_model.state_dict(), model_path)
                    logger.info(f"✓ Melhor loss: {best_loss:.6f}. Modelo salvo: {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping acionado após {epoch+1} épocas")
                        break
            
            # Carregar melhor checkpoint
            save_dir = self.paths.get('model_save_dir', 
                                     self.config.get('model_save_dir', 'resultados'))
            best_model_path = f"{save_dir}/best_{asset_name}.pt"
            if os.path.exists(best_model_path):
                self.base_model.load_state_dict(
                    torch.load(best_model_path, map_location=self.device)
                )
                logger.info(f"✓ Carregado melhor modelo: {best_model_path}")
            
            all_history[asset_name] = phase_history
            logger.info(f"✓ Fine-tuning de {asset_name} concluído. Loss Final: {best_loss:.6f}\n")
        
        self.base_model.eval()
        logger.info("="*70)
        logger.info("Fine-tuning de TODOS os ativos concluído")
        logger.info("="*70)
        
        return all_history

    def fine_tune_asset(self, asset_name: str, train_loader, val_loader, 
                       epochs: int = None, lr: float = None) -> dict:
        """
        Fine-tune um ativo específico com dataloaders fornecidos.
        
        Mantém compatibilidade com código legado que passa dataloaders.
        Internamente usa os mesmos mecanismos de fine_tune_all().
        
        Args:
            asset_name (str): Nome do ativo (ex: 'PETR4')
            train_loader: DataLoader para treino (compatibilidade legada)
            val_loader: DataLoader para validação (compatibilidade legada)
            epochs (int): Épocas. Se None, lê TRAINING_CONFIG['finetune_epochs']
            lr (float): LR. Se None, lê TRAINING_CONFIG['finetune_learning_rate']
        
        Returns:
            dict: Histórico de loss {'loss': [...], 'mae': [...]}
            
        Note:
            Esta é uma interface legada. Use fine_tune_all() para fluxo integrado.
        """
        if train_loader is None or val_loader is None:
            logger.warning(
                "train_loader/val_loader None. "
                "Use fine_tune_all() em vez disso para fluxo integrado."
            )
            return {}
        
        # --- P1.1: Ler defaults se não fornecidos ---
        if epochs is None:
            epochs = TRAINING_CONFIG.get('finetune_epochs', 10)
        if lr is None:
            lr = TRAINING_CONFIG.get('finetune_learning_rate', 1e-4)
        
        logger.warning(
            f"fine_tune_asset() é compatibilidade legada. "
            f"Prefira fine_tune_all() para fluxo padrão."
        )
        logger.info(
            f"Fine-tuning {asset_name} com dataloaders fornecidos: "
            f"epochs={epochs}, lr={lr}"
        )
        
        # Usar train_loader fornecido diretamente
        return self._fine_tune_with_dataloader(
            asset_name, train_loader, epochs, lr
        )

    def _fine_tune_with_dataloader(self, asset_name: str, train_loader, 
                                   epochs: int, lr: float) -> dict:
        """
        Helper privado para fine-tune com dataloader fornecido.
        
        Usado por fine_tune_asset() (compatibilidade legada).
        """
        try:
            if not self._freeze_pinn_layers():
                raise RuntimeError(f"Falha na validação de freezing para {asset_name}")
        except RuntimeError as e:
            logger.error(f"Erro ao congelar: {e}")
            return {}
        
        trainable_params = [p for p in self.base_model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
        
        phase_history = {'loss': [], 'mae': []}
        best_loss = float('inf')
        patience = TRAINING_CONFIG.get('finetune_patience', 10)
        patience_counter = 0
        
        for epoch in range(epochs):
            self.base_model.train()
            epoch_loss = 0.0
            epoch_mae = 0.0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                x_seq, x_phy, y_real, X_time, weights, asset_ids_batch = batch
                
                x_seq = x_seq.to(self.device)
                x_phy = x_phy.to(self.device)
                y_real = y_real.to(self.device)
                weights = weights.to(self.device)
                asset_ids_batch = asset_ids_batch.to(self.device) if asset_ids_batch is not None else None
                
                outputs = self.base_model(x_seq, x_phy, asset_ids_batch)
                y_pred = outputs['price']
                
                loss_mse = torch.mean(weights * (y_pred - y_real) ** 2)
                loss_mse.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                with torch.no_grad():
                    mae = torch.mean(torch.abs(y_pred - y_real))
                    epoch_loss += loss_mse.item()
                    epoch_mae += mae.item()
                    num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_mae = epoch_mae / num_batches
            
            phase_history['loss'].append(avg_loss)
            phase_history['mae'].append(avg_mae)
            scheduler.step()
            
            logger.info(f"{asset_name} Epoch {epoch+1}/{epochs} Loss={avg_loss:.6f} MAE={avg_mae:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return phase_history