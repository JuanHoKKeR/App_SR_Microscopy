#!/usr/bin/env python3
"""
Backend FastAPI para aplicación de superresolución
Versión modular con arquitectura limpia
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import asyncio
import uvicorn
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos propios
from models.config import MODEL_CONFIGS, get_available_models, validate_model_structure
from models.loader import model_loader
from processing.image_processor import image_processor
from utils.image_utils import image_utils

# Crear aplicación FastAPI
app = FastAPI(
    title="Microscopy Super-Resolution API",
    version="1.0.0",
    description="API para superresolución de imágenes de microscopía usando ESRGAN, SwinIR y EDSR"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos Pydantic
class ModelInfo(BaseModel):
    name: str
    type: str
    architecture: str
    input_size: int
    output_size: int
    available: bool
    checkpoint_iter: Optional[int] = None
    discriminator: Optional[str] = None

class ProcessPatchRequest(BaseModel):
    model_name: str
    x: int = 0
    y: int = 0
    width: int = 256
    height: int = 256

class SequentialUpsamplingRequest(BaseModel):
    architecture: str
    start_size: int
    target_scale: int
    x: int = 0
    y: int = 0
    width: int = 256
    height: int = 256

class ProcessResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Event handlers
@app.on_event("startup")
async def startup_event():
    """Inicialización de la aplicación"""
    logger.info("🚀 Iniciando Microscopy Super-Resolution API...")
    
    # Validar estructura de modelos
    validate_model_structure()
    
    # Cargar modelos disponibles
    logger.info("📦 Cargando modelos disponibles...")
    load_results = model_loader.load_all_available_models()
    
    loaded_count = sum(load_results.values())
    total_count = len(load_results)
    
    if loaded_count == 0:
        logger.warning("⚠️  No se cargaron modelos. Verifica la estructura de archivos.")
    else:
        logger.info(f"✅ API iniciada correctamente - {loaded_count}/{total_count} modelos cargados")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la aplicación"""
    logger.info("🛑 Cerrando Microscopy Super-Resolution API...")

# Rutas principales
@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    loaded_models = model_loader.get_loaded_models()
    memory_info = model_loader.get_memory_usage()
    
    return {
        "message": "Microscopy Super-Resolution API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": len(loaded_models),
        "device": memory_info.get("device", "unknown"),
        "endpoints": {
            "models": "/models",
            "process_patch": "/process_patch",
            "process_sequential": "/process_sequential",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(asyncio.get_event_loop().time()),
        "memory": model_loader.get_memory_usage()
    }

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Obtiene lista de todos los modelos configurados"""
    models_info = []
    loaded_models = model_loader.get_loaded_models()
    
    for model_name, config in MODEL_CONFIGS.items():
        models_info.append(ModelInfo(
            name=model_name,
            type=config["type"],
            architecture=config["architecture"],
            input_size=config["input_size"],
            output_size=config["output_size"],
            available=model_name in loaded_models,
            checkpoint_iter=config.get("checkpoint_iter"),
            discriminator=config.get("discriminator")
        ))
    
    return models_info

@app.get("/models/{architecture}")
async def get_models_by_architecture(architecture: str):
    """Obtiene modelos filtrados por arquitectura"""
    from models.config import get_models_by_architecture
    
    arch_models = get_models_by_architecture(architecture)
    if not arch_models:
        raise HTTPException(status_code=404, detail=f"Arquitectura {architecture} no encontrada")
    
    loaded_models = model_loader.get_loaded_models()
    models_info = []
    
    for model_name, config in arch_models.items():
        models_info.append({
            "name": model_name,
            "available": model_name in loaded_models,
            "input_size": config["input_size"],
            "output_size": config["output_size"],
            "scale": config["scale"]
        })
    
    return {
        "architecture": architecture,
        "models": models_info,
        "total_models": len(models_info),
        "available_models": sum(1 for m in models_info if m["available"])
    }

@app.post("/process_patch")
async def process_patch(
    file: UploadFile = File(...),
    model_name: str = "esrgan_256_512",
    x: int = 0,
    y: int = 0,
    width: int = 256,
    height: int = 256
):
    """Procesa un parche específico de la imagen"""
    
    # Validar que el modelo esté cargado
    if not model_loader.is_loaded(model_name):
        raise HTTPException(
            status_code=400, 
            detail=f"Modelo {model_name} no está disponible"
        )
    
    try:
        # Leer imagen
        contents = await file.read()
        image = image_utils.bytes_to_image(contents)
        
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo procesar la imagen")
        
        # Extraer parche
        patch = image_processor.extract_patch(image, x, y, width, height)
        
        # Procesar parche
        enhanced_patch = image_processor.process_single_patch(patch, model_name)
        
        if enhanced_patch is None:
            raise HTTPException(status_code=500, detail="Error procesando el parche")
        
        # Convertir a base64
        original_b64 = image_utils.image_to_base64(patch)
        enhanced_b64 = image_utils.image_to_base64(enhanced_patch)
        
        # Obtener información del modelo
        model_info = model_loader.get_model(model_name)
        
        return JSONResponse({
            "success": True,
            "original_patch": original_b64,
            "enhanced_patch": enhanced_b64,
            "original_size": f"{patch.shape[1]}x{patch.shape[0]}",
            "enhanced_size": f"{enhanced_patch.shape[1]}x{enhanced_patch.shape[0]}",
            "model_used": model_name,
            "architecture": model_info["config"]["architecture"],
            "scale_factor": model_info["config"]["scale"],
            "coordinates": {"x": x, "y": y, "width": width, "height": height}
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando parche: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/process_sequential")
async def process_sequential_upsampling(
    file: UploadFile = File(...),
    architecture: str = "ESRGAN",
    start_size: int = 256,
    target_scale: int = 4,
    x: int = 0,
    y: int = 0,
    width: int = 256,
    height: int = 256
):
    """Procesa upsampling secuencial para alcanzar factores de escala altos"""
    
    try:
        # Leer imagen
        contents = await file.read()
        image = image_utils.bytes_to_image(contents)
        
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo procesar la imagen")
        
        # Extraer parche
        patch = image_processor.extract_patch(image, x, y, width, height)
        
        # Redimensionar a start_size si es necesario
        if patch.shape[0] != start_size or patch.shape[1] != start_size:
            patch = image_utils.resize_image(patch, (start_size, start_size))
        
        # Procesar upsampling secuencial
        result = image_processor.process_sequential_upsampling(
            patch, start_size, target_scale, architecture
        )
        
        if not result or not result["success"]:
            raise HTTPException(
                status_code=400, 
                detail=f"No se puede alcanzar x{target_scale} con {architecture}"
            )
        
        # Preparar respuesta
        response_data = {
            "success": True,
            "architecture": architecture,
            "target_scale": target_scale,
            "upsampling_path": result["upsampling_path"],
            "original_size": result["original_size"],
            "final_size": result["final_size"],
            "steps": []
        }
        
        # Convertir imágenes de cada paso a base64
        for step in result["steps"]:
            step_data = {
                "step": step["step"],
                "model_name": step["model_name"],
                "input_size": step["input_size"],
                "output_size": step["output_size"],
                "enhanced_patch": image_utils.image_to_base64(step["enhanced_patch"])
            }
            response_data["steps"].append(step_data)
        
        # Imagen original y final
        response_data["original_patch"] = image_utils.image_to_base64(result["original_patch"])
        response_data["final_result"] = image_utils.image_to_base64(result["final_result"])
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en procesamiento secuencial: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/upsampling_path/{architecture}/{start_size}/{target_scale}")
async def get_upsampling_path(architecture: str, start_size: int, target_scale: int):
    """Obtiene la ruta de upsampling para parámetros específicos"""
    from models.config import get_upsampling_path
    
    target_size = start_size * target_scale
    path = get_upsampling_path(start_size, target_size, architecture)
    
    if not path:
        raise HTTPException(
            status_code=400,
            detail=f"No se puede alcanzar x{target_scale} desde {start_size} con {architecture}"
        )
    
    # Obtener detalles de cada paso
    path_details = []
    current_size = start_size
    
    for model_name in path:
        model_config = MODEL_CONFIGS[model_name]
        path_details.append({
            "model_name": model_name,
            "input_size": current_size,
            "output_size": current_size * 2,
            "architecture": model_config["architecture"],
            "available": model_loader.is_loaded(model_name)
        })
        current_size *= 2
    
    return {
        "architecture": architecture,
        "start_size": start_size,
        "target_scale": target_scale,
        "target_size": target_size,
        "path": path,
        "path_details": path_details,
        "steps_required": len(path),
        "all_models_available": all(detail["available"] for detail in path_details)
    }

@app.post("/load_model/{model_name}")
async def load_model(model_name: str):
    """Carga un modelo específico"""
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Modelo {model_name} no encontrado")
    
    if model_loader.is_loaded(model_name):
        return {"message": f"Modelo {model_name} ya está cargado", "success": True}
    
    success = model_loader.load_model(model_name)
    
    if success:
        return {"message": f"Modelo {model_name} cargado correctamente", "success": True}
    else:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo {model_name}")

@app.delete("/unload_model/{model_name}")
async def unload_model(model_name: str):
    """Descarga un modelo de la memoria"""
    if not model_loader.is_loaded(model_name):
        raise HTTPException(status_code=404, detail=f"Modelo {model_name} no está cargado")
    
    success = model_loader.unload_model(model_name)
    
    if success:
        return {"message": f"Modelo {model_name} descargado", "success": True}
    else:
        raise HTTPException(status_code=500, detail=f"Error descargando modelo {model_name}")

@app.get("/stats")
async def get_stats():
    """Obtiene estadísticas del sistema"""
    return {
        "processing_stats": image_processor.get_processing_stats(),
        "memory_usage": model_loader.get_memory_usage(),
        "loaded_models": list(model_loader.get_loaded_models().keys()),
        "available_models": len(get_available_models()),
        "total_configured_models": len(MODEL_CONFIGS)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )