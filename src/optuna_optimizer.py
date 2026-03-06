# /src/optuna_optimizer.py
"""
Módulo de Otimização de Hiperparâmetros com Optuna.

Faz busca bayesiana sobre MODEL_CONFIG e TRAINING_CONFIG para encontrar
a configuração que minimiza a val_loss em um treino rápido (fast trial).

Uso:
    python src/optuna_optimizer.py --n-trials 50 --epochs-per-trial 10
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from copy import deepcopy

import torch
import numpy as np

# Garante que o root do projeto está no path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError(
        "Optuna não instalado. Execute: pip install optuna"
    )

from torch.utils.data import DataLoader, random_split

from src.config import PATHS, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from src.data_loader import carregar_taxa_juros, criar_dataset_hibrido
from src.model import DeepHestonHybrid
from src.trainer import PINNTrainer
from src.logger import get_logger

logger = get_logger('PINN_Optuna')


# ==============================================================================
# 1. SEARCH SPACE
# ==============================================================================

def _suggest_model_config(trial: optuna.Trial, base: dict) -> dict:
    """Define o espaço de busca para MODEL_CONFIG."""
    cfg = deepcopy(base)
    cfg['lstm_hidden_size']   = trial.suggest_categorical('lstm_hidden_size',  [32, 64, 128])
    cfg['lstm_layers']        = trial.suggest_int('lstm_layers',               1, 3)
    cfg['lstm_dropout']       = trial.suggest_float('lstm_dropout',            0.0, 0.4, step=0.1)
    cfg['pinn_hidden_layers'] = trial.suggest_int('pinn_hidden_layers',        2, 6)
    cfg['pinn_neurons']       = trial.suggest_categorical('pinn_neurons',      [32, 64, 128, 256])
    cfg['fourier_features']   = trial.suggest_categorical('fourier_features',  [64, 128, 256])
    cfg['activation']         = trial.suggest_categorical('activation',        ['tanh', 'silu', 'gelu'])
    cfg['asset_embedding_dim']= trial.suggest_categorical('asset_embedding_dim',[2, 4, 8])
    return cfg


def _suggest_training_config(trial: optuna.Trial, base: dict, epochs_per_trial: int) -> dict:
    """Define o espaço de busca para TRAINING_CONFIG."""
    cfg = deepcopy(base)
    cfg['batch_size']      = trial.suggest_categorical('batch_size',     [512, 1024, 2048, 4096])
    cfg['weight_data']     = trial.suggest_float('weight_data',          0.5, 2.0)
    cfg['weight_pde']      = trial.suggest_float('weight_pde',           0.5, 5.0)
    cfg['warmup_epochs']   = trial.suggest_int('warmup_epochs',          3, 20)
    cfg['rampup_epochs']   = trial.suggest_int('rampup_epochs',          5, 30)
    cfg['lambda_bc']       = trial.suggest_float('lambda_bc',            1.0, 20.0, log=True)
    cfg['lambda_reg']      = trial.suggest_float('lambda_reg',           0.001, 0.5, log=True)
    cfg['lambda_feller']   = trial.suggest_float('lambda_feller',        1.0, 20.0, log=True)
    # Configuração de treino rápido por trial
    cfg['epochs_per_phase'] = epochs_per_trial
    cfg['patience']         = max(5, epochs_per_trial // 3)
    cfg['learning_rates']   = [1e-3, 5e-4]   # Apenas 2 fases rápidas por trial
    return cfg


# ==============================================================================
# 2. FUNÇÃO OBJETIVO
# ==============================================================================

class PINNOptunaObjective:
    """
    Função objetivo Optuna para o pipeline PINN-Heston.

    Recebe o dataset já processado (para não reprocessar a cada trial)
    e instancia um modelo + trainer com a configuração sugerida.
    """
    def __init__(
        self,
        train_dataset,
        val_dataset,
        data_stats: dict,
        base_model_config: dict,
        base_training_config: dict,
        epochs_per_trial: int = 5,
        device: str = 'cpu',
    ):
        self.train_dataset      = train_dataset
        self.val_dataset        = val_dataset
        self.data_stats         = data_stats
        self.base_model_cfg     = base_model_config
        self.base_training_cfg  = base_training_config
        self.epochs_per_trial   = epochs_per_trial
        self.device             = device

    def __call__(self, trial: optuna.Trial) -> float:
        """Roda um trial e retorna a val_loss mínima alcançada."""
        # 1. Configurações sugeridas
        model_cfg    = _suggest_model_config(trial, self.base_model_cfg)
        training_cfg = _suggest_training_config(trial, self.base_training_cfg, self.epochs_per_trial)
        training_cfg['device'] = self.device

        # 2. DataLoaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_cfg['batch_size'],
            shuffle=True,
            num_workers=0,      # 0 para evitar problemas de multiprocessing em trials paralelos
            pin_memory=(self.device == 'cuda'),
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=training_cfg['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device == 'cuda'),
        )

        # 3. Modelo
        try:
            model = DeepHestonHybrid(config=model_cfg, data_stats=self.data_stats)
        except Exception as e:
            logger.warning(f"Trial {trial.number}: Falha ao criar modelo — {e}. Pruning.")
            raise optuna.exceptions.TrialPruned()

        # 4. Trainer (com callback de normalização DESABILITADO para trials)
        #    Evitamos o overhead do callback em cada trial, já validado no run principal.
        trainer = PINNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            data_stats=self.data_stats,
            config=training_cfg,
        )

        # Monkey-patch: desabilita callback de normalização durante trials Optuna
        # (já validado antes da busca)
        original_validate = None
        try:
            from src import utils as _utils_mod
            original_validate = _utils_mod.validate_normalization
            _utils_mod.validate_normalization = lambda *a, **kw: None
        except Exception:
            pass

        try:
            trainer.train()
        except Exception as e:
            logger.warning(f"Trial {trial.number}: Falha no treino — {e}. Pruning.")
            if original_validate:
                from src import utils as _utils_mod
                _utils_mod.validate_normalization = original_validate
            raise optuna.exceptions.TrialPruned()
        finally:
            if original_validate:
                from src import utils as _utils_mod
                _utils_mod.validate_normalization = original_validate

        # 5. Retorna melhor val_loss do trial
        val_losses = trainer.history.get('val_loss', [])
        if not val_losses:
            raise optuna.exceptions.TrialPruned()

        best_val = min(float(v) for v in val_losses if v is not None)
        logger.info(f"Trial {trial.number:3d} | val_loss={best_val:.6f}")
        return best_val


# ==============================================================================
# 3. CLASSE PRINCIPAL OPTIMIZER
# ==============================================================================

class PINNOptunaOptimizer:
    """
    Orquestrador da otimização Optuna para o pipeline PINN-Heston.

    Carrega os dados uma única vez, instancia o estudo Optuna e
    salva os resultados em resultados/optuna_best_config_{timestamp}.txt.
    """
    def __init__(
        self,
        n_trials: int = 50,
        epochs_per_trial: int = 5,
        results_dir: str = None,
    ):
        self.n_trials        = n_trials
        self.epochs_per_trial = epochs_per_trial
        self.results_dir     = results_dir or PATHS['results_dir']
        self.device          = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"PINNOptunaOptimizer | device={self.device} | n_trials={n_trials} | epochs/trial={epochs_per_trial}")

    def _load_data(self):
        """Carrega e processa o dataset uma única vez."""
        logger.info("Carregando dados para otimização...")
        df_juros = carregar_taxa_juros(PATHS['selic_data'])
        if df_juros is None:
            raise RuntimeError("Falha ao carregar taxa Selic.")

        full_dataset, data_stats = criar_dataset_hibrido(
            caminho_pasta_opcoes=PATHS['raw_data'],
            df_juros=df_juros,
            seq_length=DATA_CONFIG['sequence_length'],
        )
        if full_dataset is None:
            raise RuntimeError("Dataset vazio após processamento.")

        # Atualiza num_assets dinamicamente
        base_model_cfg = deepcopy(MODEL_CONFIG)
        n_assets = len(data_stats.get('asset_map', {}))
        base_model_cfg['num_assets'] = max(n_assets, 10)

        # Split treino / validação
        total      = len(full_dataset)
        val_size   = int(total * DATA_CONFIG['test_size'])
        train_size = total - val_size
        generator  = torch.Generator().manual_seed(DATA_CONFIG['random_state'])
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

        logger.info(f"Dataset OK: {train_size} treino | {val_size} validação | {n_assets} ativos")
        return train_ds, val_ds, data_stats, base_model_cfg

    def _save_results(self, study: optuna.Study, elapsed_minutes: float):
        """Salva relatório detalhado em .txt na pasta resultados."""
        os.makedirs(self.results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = os.path.join(self.results_dir, f"optuna_best_config_{timestamp}.txt")

        best  = study.best_trial
        trials = study.trials

        lines = [
            "=" * 70,
            "  OPTUNA HYPERPARAMETER OPTIMIZATION — PINN Heston-LSTM Híbrido",
            "=" * 70,
            f"  Data/Hora       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Trials Totais   : {len(trials)}",
            f"  Trials Completos: {len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE])}",
            f"  Trials Podados  : {len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED])}",
            f"  Tempo Total     : {elapsed_minutes:.1f} min",
            f"  Dispositivo     : {self.device}",
            f"  Epochs/Trial    : {self.epochs_per_trial}",
            "",
            "─" * 70,
            "  MELHOR TRIAL",
            "─" * 70,
            f"  Trial #         : {best.number}",
            f"  val_loss        : {best.value:.6f}",
            "",
            "  ── MODEL_CONFIG (hiperparâmetros) ──",
        ]

        model_keys = [
            'lstm_hidden_size', 'lstm_layers', 'lstm_dropout',
            'pinn_hidden_layers', 'pinn_neurons', 'fourier_features',
            'activation', 'asset_embedding_dim'
        ]
        for k in model_keys:
            if k in best.params:
                lines.append(f"    '{k}': {best.params[k]},")

        lines += ["", "  ── TRAINING_CONFIG (hiperparâmetros) ──"]
        training_keys = [
            'batch_size', 'weight_data', 'weight_pde',
            'warmup_epochs', 'rampup_epochs',
            'lambda_bc', 'lambda_reg', 'lambda_feller'
        ]
        for k in training_keys:
            if k in best.params:
                lines.append(f"    '{k}': {best.params[k]:.6g},")

        lines += [
            "",
            "─" * 70,
            "  TOP-10 TRIALS (por val_loss crescente)",
            "─" * 70,
        ]
        completed = sorted(
            [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value
        )
        for rank, t in enumerate(completed[:10], 1):
            lines.append(f"  #{rank:2d}  Trial={t.number:3d}  val_loss={t.value:.6f}")

        lines += [
            "",
            "─" * 70,
            "  TODOS OS PARÂMETROS DO MELHOR TRIAL (para copiar em config.py)",
            "─" * 70,
            "  MODEL_CONFIG = {",
        ]
        for k in model_keys:
            if k in best.params:
                v = best.params[k]
                lines.append(f"      '{k}': {repr(v)},")
        lines += ["      # ... (manter outros parâmetros de MODEL_CONFIG) ..."]
        lines += ["  }", "", "  TRAINING_CONFIG = {"]
        for k in training_keys:
            if k in best.params:
                v = best.params[k]
                lines.append(f"      '{k}': {repr(v)},")
        lines += [
            "      # ... (manter outros parâmetros de TRAINING_CONFIG) ...",
            "  }",
            "",
            "=" * 70,
        ]

        content = "\n".join(lines)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Resultados Optuna salvos em: {out_path}")
        print(content)  # Exibe no terminal também
        return out_path

    def run(self) -> optuna.Study:
        """Executa a otimização completa."""
        train_ds, val_ds, data_stats, base_model_cfg = self._load_data()

        objective = PINNOptunaObjective(
            train_dataset     = train_ds,
            val_dataset       = val_ds,
            data_stats        = data_stats,
            base_model_config = base_model_cfg,
            base_training_config = deepcopy(TRAINING_CONFIG),
            epochs_per_trial  = self.epochs_per_trial,
            device            = self.device,
        )

        # Sampler TPE com seed para reprodutibilidade
        sampler = TPESampler(seed=DATA_CONFIG['random_state'])

        # Pruner Median (corta trials ruins na metade)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name="PINN_Heston_HPO",
        )

        logger.info(f"Iniciando otimização Optuna com {self.n_trials} trials...")
        t0 = time.time()

        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                catch=(Exception,),
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            logger.warning("Otimização interrompida pelo usuário.")

        elapsed = (time.time() - t0) / 60
        self._save_results(study, elapsed)

        logger.info(f"Melhor val_loss: {study.best_value:.6f}")
        logger.info(f"Melhores parâmetros: {study.best_params}")
        return study


# ==============================================================================
# 4. CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Otimização de Hiperparâmetros PINN com Optuna"
    )
    parser.add_argument(
        '--n-trials', type=int, default=50,
        help='Número de trials Optuna (default: 50)'
    )
    parser.add_argument(
        '--epochs-per-trial', type=int, default=5,
        help='Épocas de treino por trial (default: 5; mais épocas = mais preciso, mais lento)'
    )
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help='Diretório para salvar resultados (default: resultados/)'
    )
    parser.add_argument(
        '--log-level', type=str, default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Nível de log do Optuna (default: WARNING para minimizar output)'
    )
    args = parser.parse_args()

    # Silencia logs do Optuna para não poluir o output
    optuna.logging.set_verbosity(
        getattr(logging, args.log_level)
    )

    optimizer = PINNOptunaOptimizer(
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        results_dir=args.results_dir,
    )
    optimizer.run()


if __name__ == '__main__':
    main()
