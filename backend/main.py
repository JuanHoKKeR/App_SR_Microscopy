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
import numpy as np

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
from utils.quality_evaluator import quality_evaluator

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

class EvaluateQualityRequest(BaseModel):
    calculate_perceptual: bool = True

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

@app.get("/models/specific/{architecture}")
async def get_specific_models(architecture: str):
    """Obtiene modelos específicos de una arquitectura con detalles completos"""
    from models.config import get_models_by_architecture
    
    arch_models = get_models_by_architecture(architecture)
    if not arch_models:
        raise HTTPException(status_code=404, detail=f"Arquitectura {architecture} no encontrada")
    
    loaded_models = model_loader.get_loaded_models()
    detailed_models = []
    
    for model_name, config in arch_models.items():
        model_detail = {
            "name": model_name,
            "display_name": f"{architecture} {config['input_size']}→{config['output_size']}",
            "input_size": config["input_size"],
            "output_size": config["output_size"],
            "scale": config["scale"],
            "available": model_name in loaded_models,
            "checkpoint_iter": config.get("checkpoint_iter"),
            "discriminator": config.get("discriminator"),
            "optimized": config.get("optimized", False),
            "description": f"Modelo {architecture} entrenado para {config['input_size']}x{config['input_size']} → {config['output_size']}x{config['output_size']} (x{config['scale']})"
        }
        detailed_models.append(model_detail)
    
    # Ordenar por input_size
    detailed_models.sort(key=lambda x: x["input_size"])
    
    return {
        "architecture": architecture,
        "models": detailed_models,
        "total_models": len(detailed_models),
        "available_models": sum(1 for m in detailed_models if m["available"])
    }


@app.get("/patch/validate_selection")
async def validate_patch_selection(
    x: int, y: int, width: int, height: int,
    image_width: int, image_height: int,
    target_size: int = None
):
    """Valida y ajusta automáticamente la selección de parche"""
    
    # Validar que la selección esté dentro de la imagen
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    
    # Si se especifica target_size, ajustar automáticamente
    if target_size:
        # Centrar el parche en la selección original
        center_x = x + width // 2
        center_y = y + height // 2
        
        # Calcular nuevas coordenadas centradas
        new_x = max(0, min(center_x - target_size // 2, image_width - target_size))
        new_y = max(0, min(center_y - target_size // 2, image_height - target_size))
        
        adjusted_selection = {
            "x": new_x,
            "y": new_y,
            "width": target_size,
            "height": target_size,
            "auto_adjusted": True,
            "original_selection": {"x": x, "y": y, "width": width, "height": height}
        }
    else:
        # Ajustar a cuadrado usando el menor lado
        size = min(width, height, image_width - x, image_height - y)
        adjusted_selection = {
            "x": x,
            "y": y,
            "width": size,
            "height": size,
            "auto_adjusted": width != size or height != size,
            "original_selection": {"x": x, "y": y, "width": width, "height": height}
        }
    
    # Validar que el parche ajustado esté completamente dentro de la imagen
    final_x = adjusted_selection["x"]
    final_y = adjusted_selection["y"]
    final_size = adjusted_selection["width"]
    
    if final_x + final_size > image_width:
        final_x = image_width - final_size
    if final_y + final_size > image_height:
        final_y = image_height - final_size
    
    adjusted_selection.update({"x": final_x, "y": final_y})
    
    # Determinar tamaños de modelo disponibles para esta selección
    available_sizes = [64, 128, 256, 512]
    valid_sizes = [size for size in available_sizes if size <= final_size]
    
    return {
        "selection": adjusted_selection,
        "valid_model_sizes": valid_sizes,
        "recommended_size": max(valid_sizes) if valid_sizes else None,
        "can_process": len(valid_sizes) > 0
    }

@app.post("/patch/preview")
async def preview_patch(
    file: UploadFile = File(...),
    x: int = 0,
    y: int = 0,
    width: int = 256,
    height: int = 256,
    target_size: int = None
):
    """Genera preview del parche seleccionado"""
    try:
        # Leer imagen
        contents = await file.read()
        image = image_utils.bytes_to_image(contents)
        
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo procesar la imagen")
        
        # Validar selección automáticamente
        validation = await validate_patch_selection(
            x, y, width, height,
            image.shape[1], image.shape[0],
            target_size
        )
        
        if not validation["can_process"]:
            raise HTTPException(
                status_code=400, 
                detail="Selección demasiado pequeña para procesar"
            )
        
        selection = validation["selection"]
        
        # Extraer parche
        patch = image_processor.extract_patch(
            image, 
            selection["x"], 
            selection["y"], 
            selection["width"], 
            selection["height"]
        )
        
        # Redimensionar si es necesario
        if target_size and target_size != selection["width"]:
            patch = image_utils.resize_image(patch, (target_size, target_size))
        
        # Convertir a base64
        patch_b64 = image_utils.image_to_base64(patch)
        
        return {
            "success": True,
            "patch_preview": patch_b64,
            "selection": selection,
            "patch_size": f"{patch.shape[1]}x{patch.shape[0]}",
            "validation": validation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando preview: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/process_patch_advanced")
async def process_patch_advanced(
    file: UploadFile = File(...),
    model_name: str = "esrgan_256_512",
    x: int = 0,
    y: int = 0,
    width: int = 256,
    height: int = 256,
    target_size: int = None,
    evaluate_quality: bool = False,
    compare_architectures: bool = False
):
    """Procesamiento avanzado de parches con comparativas y métricas"""
    
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
        
        # Validar selección
        validation = await validate_patch_selection(
            x, y, width, height,
            image.shape[1], image.shape[0],
            target_size
        )
        
        selection = validation["selection"]
        
        # Extraer parche original
        original_patch = image_processor.extract_patch(
            image, 
            selection["x"], 
            selection["y"], 
            selection["width"], 
            selection["height"]
        )
        
        # Redimensionar para el modelo si es necesario
        model_info = model_loader.get_model(model_name)
        expected_size = model_info["config"]["input_size"]
        
        if original_patch.shape[0] != expected_size:
            input_patch = image_utils.resize_image(
                original_patch, 
                (expected_size, expected_size)
            )
        else:
            input_patch = original_patch.copy()
        
        # Procesar parche
        enhanced_patch = image_processor.process_single_patch(input_patch, model_name)
        
        if enhanced_patch is None:
            raise HTTPException(status_code=500, detail="Error procesando el parche")
        
        # Preparar respuesta
        response_data = {
            "success": True,
            "model_used": model_name,
            "model_info": {
                "architecture": model_info["config"]["architecture"],
                "input_size": model_info["config"]["input_size"],
                "output_size": model_info["config"]["output_size"],
                "scale_factor": model_info["config"]["scale"]
            },
            "selection_info": selection,
            "original_patch": image_utils.image_to_base64(original_patch),
            "input_patch": image_utils.image_to_base64(input_patch),
            "enhanced_patch": image_utils.image_to_base64(enhanced_patch),
            "sizes": {
                "original": f"{original_patch.shape[1]}x{original_patch.shape[0]}",
                "input": f"{input_patch.shape[1]}x{input_patch.shape[0]}",
                "enhanced": f"{enhanced_patch.shape[1]}x{enhanced_patch.shape[0]}"
            }
        }
        
        # Evaluación de calidad
        if evaluate_quality:
            try:
                # Redimensionar original al tamaño del enhanced para comparación
                original_resized = image_utils.resize_image(
                    original_patch,
                    (enhanced_patch.shape[1], enhanced_patch.shape[0])
                )
                
                quality_results = quality_evaluator.evaluate_image_quality(
                    original=original_resized,
                    enhanced=enhanced_patch,
                    calculate_perceptual=True
                )
                
                if quality_results["evaluation_success"]:
                    interpretation = quality_evaluator.get_quality_interpretation(quality_results)
                    response_data["quality_metrics"] = {
                        "psnr": quality_results["psnr"],
                        "ssim": quality_results["ssim"],
                        "perceptual_index": quality_results["perceptual_index"],
                        "interpretation": interpretation,
                        "kimianet_used": quality_evaluator.is_kimianet_available(),
                        "comparison_method": "original_upscaled_vs_enhanced"
                    }
                
            except Exception as e:
                logger.warning(f"Error en evaluación de calidad: {e}")
                response_data["quality_metrics"] = {"error": str(e)}
        
        # Comparación con otras arquitecturas (opcional)
        if compare_architectures:
            response_data["architecture_comparison"] = await _compare_architectures(
                input_patch, model_info["config"]
            )
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en procesamiento avanzado: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


async def _compare_architectures(input_patch: np.ndarray, reference_config: dict):
    """Compara resultado con otras arquitecturas disponibles"""
    comparisons = []
    input_size = reference_config["input_size"]
    output_size = reference_config["output_size"]
    
    # Buscar modelos equivalentes en otras arquitecturas
    for model_name, config in MODEL_CONFIGS.items():
        if (config["input_size"] == input_size and 
            config["output_size"] == output_size and
            config["architecture"] != reference_config["architecture"] and
            model_loader.is_loaded(model_name)):
            
            try:
                # Procesar con modelo alternativo
                alt_result = image_processor.process_single_patch(input_patch, model_name)
                if alt_result is not None:
                    comparisons.append({
                        "architecture": config["architecture"],
                        "model_name": model_name,
                        "result": image_utils.image_to_base64(alt_result),
                        "available": True
                    })
            except Exception as e:
                logger.warning(f"Error comparando con {model_name}: {e}")
    
    return comparisons


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
    height: int = 256,
    evaluate_quality: bool = False
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
        original_patch = patch.copy()  # Guardar original para evaluación
        
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
        
        # Preparar respuesta base
        response_data = {
            "success": True,
            "architecture": architecture,
            "target_scale": target_scale,
            "upsampling_path": result["upsampling_path"],
            "original_size": result["original_size"],
            "final_size": result["final_size"],
            "steps": [],
            "quality_metrics": None
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
        
        # Evaluación de calidad opcional
        if evaluate_quality and target_scale >= 2:
            try:
                # Para evaluación, redimensionar original al tamaño final para comparación
                final_result = result["final_result"]
                original_resized = image_utils.resize_image(
                    original_patch, 
                    (final_result.shape[1], final_result.shape[0])
                )
                
                quality_results = quality_evaluator.evaluate_image_quality(
                    original=original_resized,
                    enhanced=final_result,
                    calculate_perceptual=True
                )
                
                if quality_results["evaluation_success"]:
                    interpretation = quality_evaluator.get_quality_interpretation(quality_results)
                    response_data["quality_metrics"] = {
                        "psnr": quality_results["psnr"],
                        "ssim": quality_results["ssim"],
                        "perceptual_index": quality_results["perceptual_index"],
                        "interpretation": interpretation,
                        "kimianet_used": quality_evaluator.is_kimianet_available()
                    }
                    logger.info(f"Evaluación de calidad completada - PSNR: {quality_results['psnr']:.4f}")
                
            except Exception as e:
                logger.warning(f"Error en evaluación de calidad: {e}")
                response_data["quality_metrics"] = {"error": str(e)}
        
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
        "total_configured_models": len(MODEL_CONFIGS),
        "kimianet_available": quality_evaluator.is_kimianet_available()
    }

@app.post("/evaluate_qualitative")
async def evaluate_qualitative(
    original_file: UploadFile = File(...),
    enhanced_file: UploadFile = File(...),
    include_comparison_plot: bool = True
):
    """Evaluación cualitativa completa con análisis visual de diferencias"""
    
    try:
        # Leer imágenes
        original_contents = await original_file.read()
        enhanced_contents = await enhanced_file.read()
        
        original_image = image_utils.bytes_to_image(original_contents)
        enhanced_image = image_utils.bytes_to_image(enhanced_contents)
        
        if original_image is None or enhanced_image is None:
            raise HTTPException(status_code=400, detail="No se pudieron procesar las imágenes")
        
        # Evaluación completa usando el evaluador mejorado
        results = quality_evaluator.evaluate_image_quality_comprehensive(
            original=original_image,
            enhanced=enhanced_image,
            calculate_perceptual=True,
            calculate_qualitative=True
        )
        
        if not results["evaluation_success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"Error en evaluación: {results.get('error_message', 'Error desconocido')}"
            )
        
        # Obtener interpretación completa
        interpretation = quality_evaluator.get_quality_interpretation_comprehensive(results)
        
        # Preparar respuesta
        response_data = {
            "success": True,
            "metrics": results["metrics"],
            "interpretation": interpretation,
            "qualitative_analysis": results.get("qualitative_analysis", {}),
            "kimianet_used": quality_evaluator.is_kimianet_available(),
            "evaluation_method": "comprehensive_with_qualitative",
            "original_image_info": image_utils.get_image_info(original_image),
            "enhanced_image_info": image_utils.get_image_info(enhanced_image)
        }
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en evaluación cualitativa: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/models/recommendation")
async def get_model_recommendation(
    image_width: int,
    image_height: int,
    desired_scale: int = 2,
    architecture_preference: str = None
):
    """Recomienda el mejor modelo para una imagen específica"""
    
    try:
        # Determinar el tamaño de parche óptimo
        min_dimension = min(image_width, image_height)
        
        # Tamaños de parche disponibles (en orden de preferencia)
        available_patch_sizes = [512, 256, 128, 64]
        optimal_patch_size = None
        
        for size in available_patch_sizes:
            if min_dimension >= size:
                optimal_patch_size = size
                break
        
        if optimal_patch_size is None:
            raise HTTPException(
                status_code=400,
                detail=f"Imagen demasiado pequeña (mínimo 64x64). Tamaño actual: {image_width}x{image_height}"
            )
        
        # Buscar modelos disponibles para el tamaño óptimo
        suitable_models = []
        loaded_models = model_loader.get_loaded_models()
        
        for model_name, config in MODEL_CONFIGS.items():
            if (config["input_size"] == optimal_patch_size and 
                model_name in loaded_models):
                
                # Calcular si puede alcanzar el factor de escala deseado
                current_scale = config["scale"]
                total_scale_possible = current_scale
                
                # Verificar upsampling secuencial
                current_size = config["output_size"]
                while current_size < optimal_patch_size * desired_scale:
                    next_model = None
                    for next_name, next_config in MODEL_CONFIGS.items():
                        if (next_config["input_size"] == current_size and
                            next_config["architecture"] == config["architecture"] and
                            next_name in loaded_models):
                            next_model = next_config
                            break
                    
                    if next_model:
                        total_scale_possible *= next_model["scale"]
                        current_size = next_model["output_size"]
                    else:
                        break
                
                # Agregar a recomendaciones si es adecuado
                model_recommendation = {
                    "model_name": model_name,
                    "architecture": config["architecture"],
                    "input_size": config["input_size"],
                    "output_size": config["output_size"],
                    "scale": config["scale"],
                    "can_achieve_desired_scale": total_scale_possible >= desired_scale,
                    "max_scale_possible": total_scale_possible,
                    "recommended_for_image": True,
                    "efficiency_score": _calculate_efficiency_score(config, optimal_patch_size, desired_scale)
                }
                
                suitable_models.append(model_recommendation)
        
        # Ordenar por puntuación de eficiencia
        suitable_models.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        # Filtrar por preferencia de arquitectura si se especifica
        if architecture_preference:
            preferred_models = [m for m in suitable_models if m["architecture"].lower() == architecture_preference.lower()]
            if preferred_models:
                suitable_models = preferred_models
        
        # Preparar respuesta
        recommendation = {
            "image_dimensions": f"{image_width}x{image_height}",
            "optimal_patch_size": optimal_patch_size,
            "desired_scale": desired_scale,
            "recommended_models": suitable_models[:5],  # Top 5
            "best_model": suitable_models[0] if suitable_models else None,
            "total_suitable_models": len(suitable_models)
        }
        
        if not suitable_models:
            recommendation["warning"] = "No se encontraron modelos adecuados para estos parámetros"
        
        return recommendation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando recomendación: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

def _calculate_efficiency_score(config, patch_size, desired_scale):
    """Calcula puntuación de eficiencia para un modelo"""
    score = 0
    
    # Bonus por tamaño de entrada óptimo
    if config["input_size"] == patch_size:
        score += 50
    
    # Bonus por factor de escala
    if config["scale"] == desired_scale:
        score += 30
    elif config["scale"] == 2 and desired_scale % 2 == 0:
        score += 20  # Puede ser usado secuencialmente
    
    # Bonus por arquitectura (basado en rendimiento general)
    arch_scores = {"ESRGAN": 25, "SwinIR": 20, "EDSR": 15}
    score += arch_scores.get(config["architecture"], 10)
    
    # Bonus por características especiales
    if config.get("discriminator") == "KimiaNet":
        score += 15  # Especializado en histopatología
    
    if config.get("optimized"):
        score += 10
    
    return score

@app.get("/patch/smart_suggestions")
async def get_smart_patch_suggestions(
    image_width: int,
    image_height: int,
    architecture: str = None
):
    """Sugiere parches inteligentes basados en la imagen y modelos disponibles"""
    
    try:
        suggestions = []
        loaded_models = model_loader.get_loaded_models()
        
        # Filtrar por arquitectura si se especifica
        available_models = MODEL_CONFIGS.items()
        if architecture:
            available_models = [(name, config) for name, config in available_models 
                              if config["architecture"].lower() == architecture.lower()]
        
        # Generar sugerencias para cada modelo disponible
        for model_name, config in available_models:
            if model_name not in loaded_models:
                continue
            
            patch_size = config["input_size"]
            
            # Verificar si la imagen es lo suficientemente grande
            if image_width >= patch_size and image_height >= patch_size:
                
                # Calcular posiciones sugeridas
                positions = []
                
                # Centro de la imagen
                center_x = (image_width - patch_size) // 2
                center_y = (image_height - patch_size) // 2
                positions.append({
                    "name": "Centro",
                    "x": center_x,
                    "y": center_y,
                    "reason": "Región central, típicamente rica en detalles"
                })
                
                # Esquinas si hay espacio suficiente
                if image_width >= patch_size * 1.5 and image_height >= patch_size * 1.5:
                    positions.extend([
                        {
                            "name": "Superior Izquierda",
                            "x": 0,
                            "y": 0,
                            "reason": "Esquina superior para análisis de bordes"
                        },
                        {
                            "name": "Superior Derecha", 
                            "x": image_width - patch_size,
                            "y": 0,
                            "reason": "Esquina superior derecha"
                        },
                        {
                            "name": "Inferior Izquierda",
                            "x": 0,
                            "y": image_height - patch_size,
                            "reason": "Esquina inferior izquierda"
                        },
                        {
                            "name": "Inferior Derecha",
                            "x": image_width - patch_size,
                            "y": image_height - patch_size,
                            "reason": "Esquina inferior derecha"
                        }
                    ])
                
                suggestion = {
                    "model_name": model_name,
                    "architecture": config["architecture"],
                    "patch_size": patch_size,
                    "scale": config["scale"],
                    "suggested_positions": positions,
                    "max_patches_horizontal": image_width // patch_size,
                    "max_patches_vertical": image_height // patch_size,
                    "coverage_percentage": (patch_size * patch_size) / (image_width * image_height) * 100
                }
                
                suggestions.append(suggestion)
        
        # Ordenar por tamaño de parche (más grande primero)
        suggestions.sort(key=lambda x: x["patch_size"], reverse=True)
        
        return {
            "image_dimensions": f"{image_width}x{image_height}",
            "total_suggestions": len(suggestions),
            "suggestions": suggestions,
            "recommendation": "Use el parche más grande disponible para mejor calidad" if suggestions else "No hay modelos compatibles"
        }
        
    except Exception as e:
        logger.error(f"Error generando sugerencias: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/kimianet_status")
async def get_kimianet_status():
    """Obtiene el estado de KimiaNet"""
    is_available = quality_evaluator.is_kimianet_available()
    
    status_info = {
        "available": is_available,
        "weights_path": quality_evaluator.kimianet_weights_path if quality_evaluator.kimianet_evaluator else None,
        "model_loaded": quality_evaluator.kimianet_evaluator is not None
    }
    
    if is_available:
        status_info["message"] = "KimiaNet está disponible para evaluación perceptual"
    else:
        status_info["message"] = "KimiaNet no está disponible - verificar pesos del modelo"
    
    return status_info

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )