# /src/logger.py

"""
Sistema de Logging Centralizado para PINN Heston
Configuração unificada de logging para todo o projeto
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = 'PINN_Heston', log_dir: str = None, level: int = logging.INFO):
    """
    Configura e retorna um logger com handlers para console e arquivo.
    
    Args:
        name: Nome do logger
        log_dir: Diretório para salvar logs (None = apenas console)
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Evita duplicação de handlers se já configurado
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False
    
    # Formato detalhado para arquivo, simples para console
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    
    # Handler para console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se especificado)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f'training_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Arquivo captura tudo
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file criado: {log_file}")
    
    return logger


def get_logger(name: str = 'PINN_Heston'):
    """
    Retorna um logger já configurado ou cria um novo se não existir.
    
    Args:
        name: Nome do logger
    
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Se não tem handlers, configura com defaults
    if not logger.handlers:
        return setup_logger(name)
    
    return logger
