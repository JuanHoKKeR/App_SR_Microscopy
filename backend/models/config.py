"""
Configuración de modelos de superresolución
Adaptado a la estructura real del proyecto
"""

import os
from pathlib import Path
from typing import Dict, Any

# Ruta base de modelos
BASE_MODELS_PATH = Path("../models")

# Configuración de modelos basada en tu estructura real
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ESRGAN Models (TensorFlow SavedModel)
    "esrgan_64_128": {
        "type": "tensorflow",
        "path": BASE_MODELS_PATH / "ESRGAN/ESRGAN_64to128_KimiaNet",
        "input_size": 64,
        "output_size": 128,
        "scale": 2,
        "architecture": "ESRGAN",
        "discriminator": "KimiaNet"
    },
    "esrgan_128_256": {
        "type": "tensorflow",
        "path": BASE_MODELS_PATH / "ESRGAN/ESRGAN_128to256_KimiaNet",
        "input_size": 128,
        "output_size": 256,
        "scale": 2,
        "architecture": "ESRGAN", 
        "discriminator": "KimiaNet"
    },
    "esrgan_256_512": {
        "type": "tensorflow",
        "path": BASE_MODELS_PATH / "ESRGAN/ESRGAN_256to512_KimiaNet",
        "input_size": 256,
        "output_size": 512,
        "scale": 2,
        "architecture": "ESRGAN",
        "discriminator": "KimiaNet"
    },
    "esrgan_512_1024": {
        "type": "tensorflow",
        "path": BASE_MODELS_PATH / "ESRGAN/ESRGAN_512to1024_Optimized_KmiaNet",
        "input_size": 512,
        "output_size": 1024,
        "scale": 2,
        "architecture": "ESRGAN",
        "discriminator": "KimiaNet",
        "optimized": True
    },
    
    # SwinIR Models (PyTorch)
    "swinir_64_128": {
        "type": "pytorch_swinir",
        "path": BASE_MODELS_PATH / "SwinIR/SwinIR_SR_64to128/665000_G.pth",
        "input_size": 64,
        "output_size": 128,
        "scale": 2,
        "architecture": "SwinIR",
        "training_patch_size": 64,
        "checkpoint_iter": 665000
    },
    "swinir_128_256": {
        "type": "pytorch_swinir",
        "path": BASE_MODELS_PATH / "SwinIR/SwinIR_SR_128to256/615000_G.pth",
        "input_size": 128,
        "output_size": 256,
        "scale": 2,
        "architecture": "SwinIR",
        "training_patch_size": 128,
        "checkpoint_iter": 615000
    },
    "swinir_256_512": {
        "type": "pytorch_swinir",
        "path": BASE_MODELS_PATH / "SwinIR/SwinIR_SR_256to512/700000_G.pth",
        "input_size": 256,
        "output_size": 512,
        "scale": 2,
        "architecture": "SwinIR",
        "training_patch_size": 256,
        "checkpoint_iter": 700000
    },
    "swinir_512_1024": {
        "type": "pytorch_swinir", 
        "path": BASE_MODELS_PATH / "SwinIR/SwinIR_SR_512to1024/500000_G.pth",
        "input_size": 512,
        "output_size": 1024,
        "scale": 2,
        "architecture": "SwinIR",
        "training_patch_size": 512,
        "checkpoint_iter": 500000
    },
    
    # EDSR Models (PyTorch + BasicSR)
    "edsr_128_256": {
        "type": "pytorch_edsr",
        "path": BASE_MODELS_PATH / "EDSR/EDSR_Microscopy_128to256/net_g_150000.pth",
        "input_size": 128,
        "output_size": 256,
        "scale": 2,
        "architecture": "EDSR",
        "checkpoint_iter": 150000
    },
    "edsr_256_512": {
        "type": "pytorch_edsr",
        "path": BASE_MODELS_PATH / "EDSR/EDSR_Microscopy_256to512/net_g_95000.pth",
        "input_size": 256,
        "output_size": 512,
        "scale": 2,
        "architecture": "EDSR",
        "checkpoint_iter": 95000
    },
    "edsr_512_1024": {
        "type": "pytorch_edsr",
        "path": BASE_MODELS_PATH / "EDSR/EDSR_Microscopy_512to1024/net_g_190000.pth",
        "input_size": 512,
        "output_size": 1024,
        "scale": 2,
        "architecture": "EDSR",
        "checkpoint_iter": 190000
    }
}

def get_models_by_architecture(architecture: str) -> Dict[str, Dict[str, Any]]:
    """Obtiene modelos filtrados por arquitectura"""
    return {
        name: config for name, config in MODEL_CONFIGS.items() 
        if config["architecture"].lower() == architecture.lower()
    }

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Obtiene modelos que existen físicamente en el sistema"""
    available = {}
    for name, config in MODEL_CONFIGS.items():
        if os.path.exists(config["path"]):
            available[name] = config
        else:
            print(f"⚠️  Modelo no encontrado: {name} -> {config['path']}")
    
    return available

def get_upsampling_path(start_size: int, target_size: int, architecture: str) -> list:
    """
    Calcula la ruta óptima de upsampling para una arquitectura específica
    """
    arch_models = get_models_by_architecture(architecture)
    available_models = get_available_models()
    
    # Filtrar solo modelos disponibles de la arquitectura seleccionada
    valid_models = {
        name: config for name, config in arch_models.items() 
        if name in available_models
    }
    
    path = []
    current_size = start_size
    
    while current_size < target_size:
        next_size = current_size * 2
        
        # Buscar modelo que vaya de current_size a next_size
        model_found = None
        for name, config in valid_models.items():
            if config["input_size"] == current_size and config["output_size"] == next_size:
                model_found = name
                break
        
        if model_found:
            path.append(model_found)
            current_size = next_size
        else:
            break
    
    return path if current_size == target_size else []

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Obtiene información detallada de un modelo"""
    if model_name not in MODEL_CONFIGS:
        return {}
    
    config = MODEL_CONFIGS[model_name].copy()
    config["available"] = os.path.exists(config["path"])
    config["path_str"] = str(config["path"])
    
    return config

def validate_model_structure():
    """Valida que la estructura de modelos sea correcta"""
    print("🔍 Validando estructura de modelos...")
    
    architecture_counts = {"ESRGAN": 0, "SwinIR": 0, "EDSR": 0}
    available_models = get_available_models()
    
    for name, config in MODEL_CONFIGS.items():
        arch = config["architecture"]
        if name in available_models:
            architecture_counts[arch] += 1
            print(f"✅ {name} -> {config['path']}")
        else:
            print(f"❌ {name} -> {config['path']}")
    
    print(f"\n📊 Resumen:")
    for arch, count in architecture_counts.items():
        total = len(get_models_by_architecture(arch))
        print(f"   {arch}: {count}/{total} disponibles")
    
    return architecture_counts