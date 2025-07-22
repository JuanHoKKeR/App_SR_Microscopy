"""
Módulo de modelos - carga y configuración de modelos de ML
"""

from .config import MODEL_CONFIGS, get_available_models, get_models_by_architecture
from .loader import model_loader

__all__ = ['MODEL_CONFIGS', 'get_available_models', 'get_models_by_architecture', 'model_loader']
