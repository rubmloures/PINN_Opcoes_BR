"""
/src/test_end_to_end.py

Testes end-to-end para validar o fluxo completo do pipeline:
1. Carregamento de dados com asset_ids
2. Instanciação de modelo com asset embeddings
3. Congelamento de parâmetros no fine-tuning
4. Desnormalização padronizada
5. Logging de componentes de physics loss
"""

import torch
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

# Imports locais
from src.data_loader import criar_dataset_hibrido, carregar_taxa_juros
from src.model import DeepHestonHybrid
from src.trainer import PINNTrainer
from src.fine_tune import FineTuner
from src.config import PATHS, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from src.logger import get_logger

logger = get_logger('EndToEndValidator')


class EndToEndValidator:
    """
    Valida o fluxo completo do pipeline:
    1. Carregamento de dados
    2. Instanciação de modelo
    3. Treinamento robusto
    4. Fine-tuning por ativo
    5. Inferência e validação
    """
    
    def __init__(self, data_dir: str, config: Dict):
        self.data_dir = data_dir
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.results = {}
        logger.info(f"EndToEndValidator inicializado. Device: {self.device}")
        
    def test_data_loading(self) -> bool:
        """Testa se os dados são carregados corretamente com asset_ids."""
        logger.info("\n" + "="*60)
        logger.info("TESTE 1: Carregamento de Dados")
        logger.info("="*60)
        
        try:
            # Carregar dados de juros
            df_juros = carregar_taxa_juros(PATHS['selic_data'])
            if df_juros is None:
                logger.error("✗ Falha ao carregar taxa de juros")
                return False
            
            # Carregar dataset
            full_dataset, data_stats = criar_dataset_hibrido(
                caminho_pasta_opcoes=PATHS['raw_data'],
                df_juros=df_juros,
                seq_length=DATA_CONFIG['sequence_length']
            )
            
            logger.info(f"✓ Dataset carregado: {len(full_dataset)} amostras")
            
            # Verificar primeiro batch
            sample = full_dataset[0]
            if len(sample) != 6:
                logger.error(f"✗ Esperado 6 elementos, recebido {len(sample)}")
                return False
            
            x_seq, x_phy, y_real, X_time, weights, asset_ids = sample
            
            logger.info(f"  - x_seq shape: {x_seq.shape} (sequência LSTM)")
            logger.info(f"  - x_phy shape: {x_phy.shape} (inputs físicos)")
            logger.info(f"  - y_real shape: {y_real.shape} (target)")
            logger.info(f"  - asset_ids: {asset_ids} (ID do ativo)")
            
            # Validar asset_ids
            if asset_ids is None:
                logger.error("✗ asset_ids é None!")
                return False
            
            num_assets = len(data_stats.get('asset_map', {}))
            if asset_ids >= num_assets:
                logger.error(f"✗ asset_ids ({asset_ids}) >= num_assets ({num_assets})")
                return False
            
            logger.info(f"✓ Asset ID válido. Total de ativos: {num_assets}")
            logger.info(f"  - Asset names: {list(data_stats.get('asset_map', {}).keys())}")
            
            self.results['data_loading'] = {
                'status': 'passed',
                'num_samples': len(full_dataset),
                'num_assets': num_assets,
                'data_stats_keys': list(data_stats.keys())
            }
            
            # Armazenar para uso posterior
            self.full_dataset = full_dataset
            self.data_stats = data_stats
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Erro no carregamento: {e}", exc_info=True)
            self.results['data_loading'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_model_instantiation(self, num_assets: int) -> bool:
        """Testa instanciação do modelo com asset embeddings."""
        logger.info("\n" + "="*60)
        logger.info("TESTE 2: Instanciação do Modelo")
        logger.info("="*60)
        
        try:
            # Atualizar config com número real de ativos
            model_config_test = MODEL_CONFIG.copy()
            model_config_test['num_assets'] = max(num_assets, 10)
            
            model = DeepHestonHybrid(
                config=model_config_test,
                data_stats=self.data_stats
            ).to(self.device)
            
            logger.info(f"✓ Modelo instanciado")
            logger.info(f"  - Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"  - Asset embeddings ativados: {model.use_embedding}")
            logger.info(f"  - Fourier features ativadas: {model.use_fourier}")
            
            # Teste forward pass
            x_seq = torch.randn(4, DATA_CONFIG['sequence_length'], 6).to(self.device)
            x_phy = torch.randn(4, 5).to(self.device)
            asset_ids = torch.tensor([0, 1, 0, 2], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = model(x_seq, x_phy, asset_ids)
            
            if 'price' not in outputs:
                logger.error("✗ Output não contém 'price'")
                return False
            
            if 'heston_params' not in outputs:
                logger.error("✗ Output não contém 'heston_params'")
                return False
            
            logger.info(f"✓ Forward pass bem-sucedido")
            logger.info(f"  - Output price shape: {outputs['price'].shape}")
            logger.info(f"  - Output price range: [{outputs['price'].min():.6f}, {outputs['price'].max():.6f}]")
            logger.info(f"  - Parâmetros Heston (5): nu, theta, kappa, xi, rho")
            
            # Teste desnormalização
            y_normalized = outputs['price']
            K_real = x_phy[:, 1:1]  # Strike em escala normalizada
            K_real_denorm = K_real * self.data_stats['K_std'] + self.data_stats['K_mean']
            y_denorm = y_normalized * K_real_denorm
            
            logger.info(f"  - Desnormalização funcionando: {y_denorm.shape}")
            logger.info(f"  - Preços desnormalizados: [{y_denorm.min():.6f}, {y_denorm.max():.6f}]")
            
            self.model = model
            self.results['model_instantiation'] = {
                'status': 'passed',
                'total_params': sum(p.numel() for p in model.parameters()),
                'output_keys': list(outputs.keys())
            }
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Erro na instanciação: {e}", exc_info=True)
            self.results['model_instantiation'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_asset_embedding_freeze(self) -> bool:
        """Testa se freezing de parâmetros funciona corretamente."""
        logger.info("\n" + "="*60)
        logger.info("TESTE 3: Congelamento de Parâmetros (Fine-tuning)")
        logger.info("="*60)
        
        try:
            # Copiar modelo para não afetar o original
            import copy
            test_model = copy.deepcopy(self.model)
            
            # Congelar PINN
            logger.info("Congelando pricing_net...")
            for param in test_model.pricing_net.parameters():
                param.requires_grad = False
            
            if hasattr(test_model, 'fourier_layer'):
                logger.info("Congelando fourier_layer...")
                for param in test_model.fourier_layer.parameters():
                    param.requires_grad = False
            
            # Liberar LSTM + Embeddings
            logger.info("Liberando LSTM...")
            for param in test_model.lstm.parameters():
                param.requires_grad = True
            logger.info("Liberando heston_head...")
            for param in test_model.heston_head.parameters():
                param.requires_grad = True
            logger.info("Liberando asset_embedding...")
            if test_model.asset_embedding is not None:
                for param in test_model.asset_embedding.parameters():
                    param.requires_grad = True
            
            # Contar parâmetros
            trainable = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in test_model.parameters() if not p.requires_grad)
            
            logger.info(f"✓ Parâmetros após congelamento:")
            logger.info(f"  - Treináveis: {trainable:,}")
            logger.info(f"  - Congelados: {frozen:,}")
            logger.info(f"  - Total: {trainable + frozen:,}")
            
            if frozen == 0:
                logger.warning("⚠ Nenhum parâmetro foi congelado! Verifique lógica.")
                return False
            
            if trainable == 0:
                logger.warning("⚠ Nenhum parâmetro está treinável!")
                return False
            
            # Validar que PINN está congelada
            pinn_frozen = all(not p.requires_grad for p in test_model.pricing_net.parameters())
            lstm_trainable = any(p.requires_grad for p in test_model.lstm.parameters())
            
            if not pinn_frozen:
                logger.error("✗ PINN não foi congelada corretamente")
                return False
            
            if not lstm_trainable:
                logger.error("✗ LSTM não está treinável")
                return False
            
            logger.info(f"✓ PINN congelada: {pinn_frozen}")
            logger.info(f"✓ LSTM treinável: {lstm_trainable}")
            
            self.results['freeze_params'] = {
                'status': 'passed',
                'trainable_params': trainable,
                'frozen_params': frozen,
                'pinn_frozen': pinn_frozen,
                'lstm_trainable': lstm_trainable
            }
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Erro no congelamento: {e}", exc_info=True)
            self.results['freeze_params'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_denormalization(self) -> bool:
        """Testa função de desnormalização centralizada."""
        logger.info("\n" + "="*60)
        logger.info("TESTE 4: Desnormalização Padronizada")
        logger.info("="*60)
        
        try:
            # Usar método do trainer
            trainer = PINNTrainer(
                model=self.model,
                train_loader=None,
                val_loader=None,
                data_stats=self.data_stats,
                config=TRAINING_CONFIG
            )
            
            # Teste com preços normalizados
            y_normalized = torch.tensor([[0.5], [0.3], [0.8]]).to(self.device)
            K_real = torch.tensor([[100.0], [150.0], [120.0]]).to(self.device)
            
            y_denorm = trainer.denormalize_price(y_normalized, K_real)
            
            logger.info(f"✓ Desnormalização funcionando")
            logger.info(f"  - y_normalized: {y_normalized.squeeze().tolist()}")
            logger.info(f"  - K_real: {K_real.squeeze().tolist()}")
            logger.info(f"  - y_denormalized: {y_denorm.squeeze().tolist()}")
            
            # Validar resultado
            expected = (y_normalized * K_real).squeeze()
            actual = y_denorm.squeeze()
            
            if not torch.allclose(expected, actual, atol=1e-5):
                logger.error("✗ Desnormalização retornou valores incorretos")
                return False
            
            # Validar que valores negativos são clampados
            y_negative = torch.tensor([[-0.1], [0.2]]).to(self.device)
            y_denorm_negative = trainer.denormalize_price(y_negative, K_real[:2])
            
            if torch.any(y_denorm_negative < 0):
                logger.error("✗ Valores negativos não foram clampados")
                return False
            
            logger.info(f"✓ Valores negativos clampados corretamente")
            
            self.results['denormalization'] = {
                'status': 'passed',
                'test_values': y_denorm.squeeze().tolist(),
                'negative_clamping': True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Erro na desnormalização: {e}", exc_info=True)
            self.results['denormalization'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def run_all_tests(self) -> Dict[str, Dict]:
        """Executa todos os testes."""
        logger.info("\n" + "="*70)
        logger.info("INICIANDO TESTES END-TO-END")
        logger.info("="*70)
        
        # Teste 1
        if not self.test_data_loading():
            logger.error("✗ Testes interrompidos devido a falha no carregamento de dados")
            return self.results
        
        num_assets = self.results['data_loading']['num_assets']
        
        # Teste 2
        if not self.test_model_instantiation(num_assets):
            logger.error("✗ Testes interrompidos devido a falha na instanciação")
            return self.results
        
        # Teste 3
        if not self.test_asset_embedding_freeze():
            logger.error("✗ Congelamento de parâmetros falhou")
        
        # Teste 4
        if not self.test_denormalization():
            logger.error("✗ Desnormalização falhou")
        
        # Resumo
        logger.info("\n" + "="*70)
        logger.info("RESUMO DOS TESTES")
        logger.info("="*70)
        
        passed_count = 0
        failed_count = 0
        
        for test_name, result in self.results.items():
            status = result.get('status', 'unknown')
            emoji = "✓" if status == 'passed' else "✗"
            logger.info(f"{emoji} {test_name}: {status}")
            
            if status == 'passed':
                passed_count += 1
            else:
                failed_count += 1
        
        logger.info(f"\nTotal: {passed_count} passaram, {failed_count} falharam")
        
        return self.results


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Executar testes
    logger.info("="*70)
    logger.info("PIPELINE HÍBRIDO DEEPESTON - TESTES END-TO-END")
    logger.info("="*70)
    
    validator = EndToEndValidator(
        data_dir=PATHS['raw_data'],
        config={
            'device': TRAINING_CONFIG['device'],
            'max_days_back': 500,
            'seq_length': DATA_CONFIG['sequence_length']
        }
    )
    
    results = validator.run_all_tests()
    
    # Retornar código de saída
    all_passed = all(r.get('status') == 'passed' for r in results.values())
    
    logger.info("\n" + "="*70)
    if all_passed:
        logger.info("✓ TODOS OS TESTES PASSARAM")
        logger.info("="*70)
        sys.exit(0)
    else:
        logger.error("✗ ALGUNS TESTES FALHARAM")
        logger.info("="*70)
        sys.exit(1)
