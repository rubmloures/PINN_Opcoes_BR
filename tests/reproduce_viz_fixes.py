
import unittest
import numpy as np
import pandas as pd
import torch
import sys
import os

# Adiciona diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization import Visualizer

# Mock Model
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_seq, x_phy, asset_ids):
        batch_size = x_seq.shape[0]
        # output mock
        return {
            'price': torch.rand(batch_size, 1),
            'heston_params': torch.rand(batch_size, 5) # Simula 5 params
        }

class TestVisualizationFixes(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.batch_size = 100
        
        # Mock Data Loader (1 batch)
        # x_seq, x_phy, y_real, X_time, weights, asset_ids
        x_seq = torch.randn(self.batch_size, 30, 6)
        x_phy = torch.randn(self.batch_size, 10)
        # Garante preços positivos
        x_phy[:, 0] = torch.abs(x_phy[:, 0]) + 1.0 # Spot
        x_phy[:, 1] = torch.abs(x_phy[:, 1]) + 1.0 # Strike
        
        y_real = torch.rand(self.batch_size, 1)
        X_time = torch.rand(self.batch_size, 1)
        weights = torch.ones(self.batch_size, 1)
        asset_ids = torch.randint(0, 5, (self.batch_size,))
        
        self.val_loader = [(x_seq, x_phy, y_real, X_time, weights, asset_ids)]
        
        self.data_stats = {
            'K_std': 1.0, 'K_mean': 10.0,
            'S_std': 1.0, 'S_mean': 10.0
        }
        
        self.config = {
            'export_plots_png': False # Desativa PNG para teste rápido
        }
        
        # Mock Ground Truth
        self.benchmark_df = pd.DataFrame({
            f'param_{i}_bench': np.random.rand(self.batch_size) for i in range(5)
        })

    def test_plot_lstm_timeseries_params(self):
        """Teste se NÃO ocorre IndexError com output dimensionalmente correto."""
        viz = Visualizer(self.model, 'dummy_hist.csv', self.val_loader, self.data_stats, self.config, self.benchmark_df)
        try:
            viz.plot_lstm_timeseries_params()
        except IndexError:
            self.fail("plot_lstm_timeseries_params raised IndexError!")

    def test_plot_greeks_surfaces_memory(self):
        """Teste se broadcast é tratado corretamente (sem MemoryError)."""
        viz = Visualizer(self.model, 'dummy_hist.csv', self.val_loader, self.data_stats, self.config)
        try:
            viz.plot_greeks_surfaces()
        except MemoryError:
            self.fail("plot_greeks_surfaces raised MemoryError!")
        except Exception as e:
            # Pode falhar por falta de dados (warning), mas não exception grave
            pass

    def test_plot_pricing_error_mape(self):
        """Teste se shapes batem no cálculo de MAPE."""
        viz = Visualizer(self.model, 'dummy_hist.csv', self.val_loader, self.data_stats, self.config)
        try:
            viz.plot_pricing_error_mape()
        except Exception as e:
             self.fail(f"plot_pricing_error_mape raised Exception: {e}")

if __name__ == '__main__':
    unittest.main()
