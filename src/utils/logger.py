"""
Módulo para configuração de logging
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_dir: Path = None, level: str = "INFO") -> logging.Logger:
    """Configura e retorna um logger"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Formata as mensagens
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo se diretório especificado
    if log_dir:
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_dir / f"ml_comparison_{timestamp}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger