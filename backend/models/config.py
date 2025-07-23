"""
Configuración de modelos de superresolución actualizada
Mejorada para manejar selección por escala y validación robusta
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

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
    
    # EDSR Models (PyTorch + BasicSR) - Falta el 64->128
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

def get_upsampling_path(start_size: int, target_size: int, architecture: str) -> List[str]:
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

def get_scale_capabilities(architecture: str) -> Dict[str, Any]:
    """
    Obtiene las capacidades de escala para una arquitectura específica
    """
    arch_models = get_models_by_architecture(architecture)
    available_models = get_available_models()
    
    # Filtrar solo modelos disponibles
    valid_models = {
        name: config for name, config in arch_models.items() 
        if name in available_models
    }
    
    if not valid_models:
        return {
            "available_sizes": [],
            "max_scales": {},
            "supported_scales": [],
            "total_models": 0
        }
    
    # Obtener tamaños de entrada disponibles
    input_sizes = sorted(list(set([config["input_size"] for config in valid_models.values()])))
    
    # Calcular escalas máximas para cada tamaño
    max_scales = {}
    for size in input_sizes:
        max_scale = calculate_max_scale_from_size(size, valid_models)
        max_scales[size] = max_scale
    
    # Calcular escalas soportadas globalmente
    supported_scales = []
    for scale in [2, 4, 8, 16]:
        if any(max_scales.get(size, 1) >= scale for size in input_sizes):
            supported_scales.append(scale)
    
    return {
        "available_sizes": input_sizes,
        "max_scales": max_scales,
        "supported_scales": supported_scales,
        "total_models": len(valid_models),
        "architecture": architecture
    }

def calculate_max_scale_from_size(start_size: int, valid_models: Dict[str, Dict[str, Any]]) -> int:
    """Calcula la escala máxima posible desde un tamaño dado"""
    max_scale = 1
    current_size = start_size
    
    while True:
        next_size = current_size * 2
        # Buscar si existe un modelo que vaya de current_size a next_size
        model_exists = any(
            config["input_size"] == current_size and config["output_size"] == next_size 
            for config in valid_models.values()
        )
        
        if model_exists:
            max_scale *= 2
            current_size = next_size
        else:
            break
    
    return max_scale

def validate_upsampling_request(start_size: int, target_scale: int, architecture: str) -> Dict[str, Any]:
    """
    Valida una solicitud de upsampling y devuelve información detallada
    """
    target_size = start_size * target_scale
    path = get_upsampling_path(start_size, target_size, architecture)
    
    # Obtener capacidades de la arquitectura
    capabilities = get_scale_capabilities(architecture)
    
    # Verificar si el tamaño inicial está disponible
    size_available = start_size in capabilities["available_sizes"]
    
    # Verificar si la escala es soportada desde este tamaño
    max_scale_from_size = capabilities["max_scales"].get(start_size, 1)
    scale_possible = target_scale <= max_scale_from_size
    
    # Verificar que todos los modelos en la ruta estén disponibles
    available_models = get_available_models()
    all_models_available = all(model_name in available_models for model_name in path)
    
    result = {
        "valid": bool(path) and size_available and scale_possible and all_models_available,
        "path": path,
        "path_length": len(path),
        "start_size": start_size,
        "target_size": target_size,
        "target_scale": target_scale,
        "size_available": size_available,
        "scale_possible": scale_possible,
        "max_scale_from_size": max_scale_from_size,
        "all_models_available": all_models_available,
        "architecture": architecture
    }
    
    # Agregar mensaje de error si no es válido
    if not result["valid"]:
        if not size_available:
            result["error"] = f"Tamaño inicial {start_size}px no disponible para {architecture}"
        elif not scale_possible:
            result["error"] = f"Escala x{target_scale} no alcanzable desde {start_size}px (máximo: x{max_scale_from_size})"
        elif not all_models_available:
            missing_models = [m for m in path if m not in available_models]
            result["error"] = f"Modelos no disponibles: {', '.join(missing_models)}"
        else:
            result["error"] = f"No se puede procesar x{target_scale} desde {start_size}px con {architecture}"
    
    return result

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Obtiene información detallada de un modelo"""
    if model_name not in MODEL_CONFIGS:
        return {}
    
    config = MODEL_CONFIGS[model_name].copy()
    config["available"] = os.path.exists(config["path"])
    config["path_str"] = str(config["path"])
    
    return config

def get_architecture_summary() -> Dict[str, Any]:
    """Obtiene resumen de todas las arquitecturas disponibles"""
    summary = {}
    
    for architecture in ["ESRGAN", "SwinIR", "EDSR"]:
        capabilities = get_scale_capabilities(architecture)
        available_models = len([
            name for name, config in MODEL_CONFIGS.items() 
            if config["architecture"] == architecture and os.path.exists(config["path"])
        ])
        total_models = len([
            name for name, config in MODEL_CONFIGS.items() 
            if config["architecture"] == architecture
        ])
        
        summary[architecture] = {
            "available_models": available_models,
            "total_models": total_models,
            "availability_percentage": (available_models / total_models * 100) if total_models > 0 else 0,
            "available_sizes": capabilities["available_sizes"],
            "supported_scales": capabilities["supported_scales"],
            "max_scale_overall": max(capabilities["max_scales"].values()) if capabilities["max_scales"] else 1
        }
    
    return summary

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
        capabilities = get_scale_capabilities(arch)
        max_scale = max(capabilities["max_scales"].values()) if capabilities["max_scales"] else 1
        print(f"   {arch}: {count}/{total} disponibles (escala máxima: x{max_scale})")
    
    # Mostrar capacidades detalladas por arquitectura
    print(f"\n📋 Capacidades por Arquitectura:")
    summary = get_architecture_summary()
    for arch, info in summary.items():
        print(f"   {arch}:")
        print(f"     - Tamaños disponibles: {info['available_sizes']}")
        print(f"     - Escalas soportadas: {info['supported_scales']}")
        print(f"     - Escala máxima: x{info['max_scale_overall']}")
    
    return architecture_counts

# Funciones de utilidad para el endpoint de rutas
def get_upsampling_path_details(start_size: int, target_scale: int, architecture: str) -> Dict[str, Any]:
    """
    Obtiene detalles completos de la ruta de upsampling incluyendo información de cada modelo
    """
    validation = validate_upsampling_request(start_size, target_scale, architecture)
    
    if not validation["valid"]:
        return validation
    
    # Obtener detalles de cada modelo en la ruta
    path_details = []
    current_size = start_size
    
    for model_name in validation["path"]:
        model_config = MODEL_CONFIGS[model_name]
        available_models = get_available_models()
        
        path_details.append({
            "model_name": model_name,
            "input_size": current_size,
            "output_size": current_size * 2,
            "architecture": model_config["architecture"],
            "type": model_config["type"],
            "available": model_name in available_models,
            "checkpoint_iter": model_config.get("checkpoint_iter"),
            "discriminator": model_config.get("discriminator")
        })
        current_size *= 2
    
    # Agregar detalles completos al resultado de validación
    validation["path_details"] = path_details
    validation["estimated_time"] = len(path_details) * 2.5  # segundos estimados
    validation["memory_required_mb"] = (target_scale ** 2) * start_size * start_size * 3 * 4 / (1024**2)  # estimación
    
    return validation