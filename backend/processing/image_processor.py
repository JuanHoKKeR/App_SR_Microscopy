"""
Procesador de imágenes para superresolución
Maneja el procesamiento con ESRGAN, SwinIR y EDSR
"""

import numpy as np
import cv2
import torch
import tensorflow as tf
import logging
from typing import Tuple, Optional, Dict, Any, List
from models.loader import model_loader
from models.config import get_upsampling_path

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Procesador de imágenes para superresolución"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ImageProcessor inicializado - Device: {self.device}")
    
    def preprocess_esrgan(self, patch: np.ndarray) -> tf.Tensor:
        """Preprocesa imagen para ESRGAN (TensorFlow)"""
        # Convertir a float32 y expandir batch dimension
        patch_float = patch.astype(np.float32)
        patch_tensor = tf.expand_dims(patch_float, 0)
        return patch_tensor
    
    def postprocess_esrgan(self, output: tf.Tensor) -> np.ndarray:
        """Postprocesa salida de ESRGAN"""
        # Remover batch dimension y convertir a uint8
        enhanced = tf.squeeze(output, 0)
        enhanced = tf.clip_by_value(enhanced, 0, 255)
        enhanced = tf.cast(enhanced, tf.uint8)
        return enhanced.numpy()
    
    def preprocess_swinir(self, patch: np.ndarray) -> torch.Tensor:
        """Preprocesa imagen para SwinIR (PyTorch)"""
        # BGR a RGB y normalizar a [0,1]
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_float = patch_rgb.astype(np.float32) / 255.0
        
        # HWC a CHW y añadir batch dimension
        patch_tensor = torch.from_numpy(np.transpose(patch_float, (2, 0, 1))).float()
        patch_tensor = patch_tensor.unsqueeze(0).to(self.device)
        
        return patch_tensor
    
    def postprocess_swinir(self, output: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocesa salida de SwinIR"""
        scale = 2  # Asumiendo x2 scale
        h_old, w_old = original_size
        
        # Recortar padding si se aplicó
        output = output[..., :h_old * scale, :w_old * scale]
        
        # Convertir de vuelta a imagen
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        if output.ndim == 3:
            output = np.transpose(output, (1, 2, 0))  # CHW a HWC
        
        # RGB a BGR y convertir a uint8
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (output * 255.0).round().astype(np.uint8)
        
        return output
    
    def preprocess_edsr(self, patch: np.ndarray) -> torch.Tensor:
        """Preprocesa imagen para EDSR (PyTorch con BasicSR)"""
        try:
            from basicsr.utils.img_util import img2tensor
            patch_tensor = img2tensor(patch, bgr2rgb=True, float32=True)
            patch_tensor = patch_tensor.unsqueeze(0).to(self.device)
            return patch_tensor
        except ImportError:
            logger.error("BasicSR no disponible para EDSR")
            raise
    
    def postprocess_edsr(self, output: torch.Tensor) -> np.ndarray:
        """Postprocesa salida de EDSR"""
        try:
            from basicsr.utils.img_util import tensor2img
            enhanced = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 255))
            return enhanced
        except ImportError:
            logger.error("BasicSR no disponible para EDSR")
            raise
    
    def apply_window_padding(self, patch_tensor: torch.Tensor, window_size: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Aplica padding para que la imagen sea múltiplo de window_size (SwinIR)"""
        _, _, h_old, w_old = patch_tensor.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        
        if h_pad > 0 or w_pad > 0:
            patch_tensor = torch.cat([patch_tensor, torch.flip(patch_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
            patch_tensor = torch.cat([patch_tensor, torch.flip(patch_tensor, [3])], 3)[:, :, :, :w_old + w_pad]
        
        return patch_tensor, (h_old, w_old)
    
    def process_patch_esrgan(self, patch: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """Procesa parche con ESRGAN"""
        try:
            model_info = model_loader.get_model(model_name)
            if not model_info:
                logger.error(f"Modelo ESRGAN {model_name} no está cargado")
                return None
            
            model = model_info["model"]
            
            # Preprocesar
            patch_tensor = self.preprocess_esrgan(patch)
            
            # Aplicar modelo
            enhanced = model(patch_tensor)
            
            # Postprocesar
            result = self.postprocess_esrgan(enhanced)
            
            logger.info(f"✅ Parche procesado con ESRGAN {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando con ESRGAN {model_name}: {e}")
            return None
    
    def process_patch_swinir(self, patch: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """Procesa parche con SwinIR"""
        try:
            model_info = model_loader.get_model(model_name)
            if not model_info:
                logger.error(f"Modelo SwinIR {model_name} no está cargado")
                return None
            
            model = model_info["model"]
            
            # Preprocesar
            patch_tensor = self.preprocess_swinir(patch)
            
            # Aplicar padding para window_size
            patch_tensor, original_size = self.apply_window_padding(patch_tensor, window_size=8)
            
            # Aplicar modelo
            with torch.no_grad():
                enhanced = model(patch_tensor)
            
            # Postprocesar
            result = self.postprocess_swinir(enhanced, original_size)
            
            logger.info(f"✅ Parche procesado con SwinIR {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando con SwinIR {model_name}: {e}")
            return None
    
    def process_patch_edsr(self, patch: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """Procesa parche con EDSR"""
        try:
            model_info = model_loader.get_model(model_name)
            if not model_info:
                logger.error(f"Modelo EDSR {model_name} no está cargado")
                return None
            
            model = model_info["model"]
            
            # Preprocesar
            patch_tensor = self.preprocess_edsr(patch)
            
            # Aplicar modelo
            with torch.no_grad():
                enhanced = model(patch_tensor)
            
            # Postprocesar
            result = self.postprocess_edsr(enhanced)
            
            logger.info(f"✅ Parche procesado con EDSR {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando con EDSR {model_name}: {e}")
            return None
    
    def process_single_patch(self, patch: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """Procesa un parche con el modelo especificado"""
        model_info = model_loader.get_model(model_name)
        if not model_info:
            logger.error(f"Modelo {model_name} no disponible")
            return None
        
        model_type = model_info["type"]
        
        # Redimensionar parche si es necesario
        expected_size = model_info["config"]["input_size"]
        if patch.shape[0] != expected_size or patch.shape[1] != expected_size:
            patch = cv2.resize(patch, (expected_size, expected_size))
            logger.info(f"Parche redimensionado a {expected_size}x{expected_size}")
        
        # Procesar según tipo de modelo
        if model_type == "tensorflow":
            return self.process_patch_esrgan(patch, model_name)
        elif model_type == "pytorch_swinir":
            return self.process_patch_swinir(patch, model_name)
        elif model_type == "pytorch_edsr":
            return self.process_patch_edsr(patch, model_name)
        else:
            logger.error(f"Tipo de modelo desconocido: {model_type}")
            return None
    
    def process_sequential_upsampling(
        self, 
        patch: np.ndarray, 
        start_size: int, 
        target_scale: int, 
        architecture: str
    ) -> Optional[Dict[str, Any]]:
        """
        Procesa upsampling secuencial para alcanzar un factor de escala alto
        """
        target_size = start_size * target_scale
        upsampling_path = get_upsampling_path(start_size, target_size, architecture)
        
        if not upsampling_path:
            logger.error(f"No se puede alcanzar x{target_scale} desde {start_size} con {architecture}")
            return None
        
        logger.info(f"Ruta de upsampling: {' → '.join(upsampling_path)}")
        
        current_patch = patch.copy()
        results = []
        
        for i, model_name in enumerate(upsampling_path):
            logger.info(f"Paso {i+1}/{len(upsampling_path)}: Aplicando {model_name}")
            
            # Procesar con el modelo actual
            enhanced_patch = self.process_single_patch(current_patch, model_name)
            
            if enhanced_patch is None:
                logger.error(f"Error en paso {i+1} con {model_name}")
                return None
            
            # Guardar resultado del paso
            model_info = model_loader.get_model(model_name)
            results.append({
                "step": i + 1,
                "model_name": model_name,
                "input_size": f"{current_patch.shape[1]}x{current_patch.shape[0]}",
                "output_size": f"{enhanced_patch.shape[1]}x{enhanced_patch.shape[0]}",
                "enhanced_patch": enhanced_patch,
                "model_config": model_info["config"] if model_info else {}
            })
            
            # Usar resultado como entrada del siguiente paso
            current_patch = enhanced_patch
        
        logger.info(f"✅ Upsampling secuencial completado: x{target_scale}")
        
        return {
            "success": True,
            "original_patch": patch,
            "final_result": current_patch,
            "upsampling_path": upsampling_path,
            "steps": results,
            "original_size": f"{patch.shape[1]}x{patch.shape[0]}",
            "final_size": f"{current_patch.shape[1]}x{current_patch.shape[0]}",
            "scale_achieved": target_scale,
            "architecture_used": architecture
        }
    
    def extract_patch(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Extrae un parche de la imagen"""
        # Asegurar que las coordenadas estén dentro de los límites
        h, w = image.shape[:2]
        x = max(0, min(x, w - width))
        y = max(0, min(y, h - height))
        width = min(width, w - x)
        height = min(height, h - y)
        
        patch = image[y:y+height, x:x+width]
        
        logger.info(f"Parche extraído: ({x},{y}) - {width}x{height}")
        return patch
    
    def validate_patch_size(self, patch: np.ndarray, expected_size: int) -> bool:
        """Valida que el parche tenga el tamaño esperado"""
        h, w = patch.shape[:2]
        return h == expected_size and w == expected_size
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de procesamiento"""
        return {
            "device": str(self.device),
            "loaded_models": len(model_loader.get_loaded_models()),
            "memory_usage": model_loader.get_memory_usage()
        }

    def process_full_image(self, image: np.ndarray, target_scale: int, 
                        architecture: str, strategy: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Procesa imagen completa usando la estrategia óptima"""
        
        # Calcular estrategia si no se proporciona
        if strategy is None:
            strategy_info = image_utils.calculate_optimal_strategy(
                image.shape[:2], target_scale, architecture
            )
            strategy = strategy_info["recommended_strategy"]
        
        if not strategy:
            logger.error("No se pudo determinar estrategia de procesamiento")
            return None
        
        try:
            if strategy["type"] == "full_image":
                return self._process_as_single_image(image, target_scale, architecture)
            else:
                return self._process_with_patches(image, target_scale, architecture, strategy)
                
        except Exception as e:
            logger.error(f"Error en procesamiento de imagen completa: {e}")
            return None

    def _process_as_single_image(self, image: np.ndarray, target_scale: int, 
                            architecture: str) -> Dict[str, Any]:
        """Procesa imagen como una sola unidad"""
        logger.info("Procesando imagen completa sin división")
        
        # Usar procesamiento secuencial existente
        result = self.process_sequential_upsampling(
            image, min(image.shape[:2]), target_scale, architecture
        )
        
        if result and result["success"]:
            return {
                "success": True,
                "strategy": "full_image",
                "original_image": image,
                "enhanced_image": result["final_result"],
                "processing_info": result,
                "patch_count": 1
            }
        
        return {"success": False, "error": "Error en procesamiento secuencial"}

    def _process_with_patches(self, image: np.ndarray, target_scale: int,
                            architecture: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa imagen dividiendo en parches"""
        patch_size = strategy["patch_size"]
        overlap = strategy["overlap"]
        
        logger.info(f"Dividiendo imagen en parches de {patch_size}x{patch_size} con overlap {overlap}")
        
        # Dividir en parches
        patches = image_utils.split_image_for_processing(image, patch_size, overlap)
        logger.info(f"Imagen dividida en {len(patches)} parches")
        
        # Procesar cada parche
        processed_patches = []
        successful_patches = 0
        
        for i, patch_info in enumerate(patches):
            logger.info(f"Procesando parche {i+1}/{len(patches)}")
            
            # Procesar parche con upsampling secuencial
            patch_result = self.process_sequential_upsampling(
                patch_info["patch"], patch_size, target_scale, architecture
            )
            
            if patch_result and patch_result["success"]:
                patch_info["enhanced_patch"] = patch_result["final_result"]
                patch_info["processing_steps"] = patch_result["steps"]
                successful_patches += 1
            else:
                logger.warning(f"Error procesando parche {i+1}")
                # Usar interpolación bicúbica como fallback
                fallback = cv2.resize(
                    patch_info["patch"], 
                    (patch_size * target_scale, patch_size * target_scale),
                    interpolation=cv2.INTER_CUBIC
                )
                patch_info["enhanced_patch"] = fallback
                patch_info["processing_steps"] = []
            
            processed_patches.append(patch_info)
        
        # Reconstruir imagen
        logger.info("Reconstruyendo imagen desde parches procesados")
        reconstructed = image_utils.reconstruct_image_from_patches(
            processed_patches, image.shape[:2], target_scale, overlap
        )
        
        return {
            "success": True,
            "strategy": "patch_based",
            "original_image": image,
            "enhanced_image": reconstructed,
            "patch_count": len(patches),
            "successful_patches": successful_patches,
            "strategy_info": strategy,
            "patches": processed_patches if len(patches) <= 4 else []  # Solo incluir parches si son pocos
        }

    def get_processing_strategies(self, image_shape: Tuple[int, int], 
                                target_scale: int, architecture: str) -> Dict[str, Any]:
        """Obtiene estrategias disponibles para procesar imagen"""
        return image_utils.calculate_optimal_strategy(image_shape, target_scale, architecture)




# Instancia global del procesador
image_processor = ImageProcessor()