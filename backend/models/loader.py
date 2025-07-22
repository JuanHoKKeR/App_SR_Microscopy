"""
Cargador de modelos de superresolución
Maneja la carga de ESRGAN, SwinIR y EDSR
"""

import torch
import tensorflow as tf
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from .config import MODEL_CONFIGS

logger = logging.getLogger(__name__)

class ModelLoader:
    """Cargador universal de modelos de superresolución"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        logger.info(f"ModelLoader inicializado - Device: {self.device}")
    
    def load_esrgan_model(self, model_path: Path) -> Optional[Any]:
        """Carga modelo ESRGAN (TensorFlow SavedModel)"""
        try:
            logger.info(f"Cargando ESRGAN desde: {model_path}")
            model = tf.saved_model.load(str(model_path))
            logger.info("✅ ESRGAN cargado correctamente")
            return model
        except Exception as e:
            logger.error(f"❌ Error cargando ESRGAN: {e}")
            return None
    
    def load_swinir_model(self, model_path: Path, config: Dict[str, Any]) -> Optional[torch.nn.Module]:
        """Carga modelo SwinIR (PyTorch)"""
        try:
            logger.info(f"Cargando SwinIR desde: {model_path}")
            
            # Importar arquitectura SwinIR
            try:
                from models.network_swinir import SwinIR as net
            except ImportError:
                logger.error("❌ No se pudo importar network_swinir. Asegúrate de que esté en models/")
                return None
            
            # Configurar modelo según training_patch_size
            training_patch_size = config.get("training_patch_size", 256)
            scale = config.get("scale", 2)
            
            if training_patch_size == 64:
                model = net(
                    upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                    upsampler='pixelshuffle', resi_connection='1conv'
                )
            elif training_patch_size == 128:
                model = net(
                    upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=180, 
                    num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                    upsampler='pixelshuffle', resi_connection='1conv'
                )
            elif training_patch_size == 256:
                model = net(
                    upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=180, 
                    num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                    upsampler='pixelshuffle', resi_connection='1conv'
                )
            elif training_patch_size == 512:
                model = net(
                    upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[4, 4, 4, 4], embed_dim=60, 
                    num_heads=[4, 4, 4, 4], mlp_ratio=2, 
                    upsampler='pixelshuffle', resi_connection='1conv'
                )
            else:
                # Configuración por defecto
                model = net(
                    upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                    upsampler='pixelshuffle', resi_connection='1conv'
                )
            
            # Cargar pesos
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Manejar diferentes formatos de checkpoint
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
                logger.info("Usando parámetros EMA")
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
                logger.info("Usando parámetros regulares")
            else:
                state_dict = checkpoint
                logger.info("Usando state_dict directamente")
            
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model = model.to(self.device)
            
            logger.info("✅ SwinIR cargado correctamente")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error cargando SwinIR: {e}")
            return None
    
    def load_edsr_model(self, model_path: Path, config: Dict[str, Any]) -> Optional[torch.nn.Module]:
        """Carga modelo EDSR (PyTorch con BasicSR)"""
        try:
            logger.info(f"Cargando EDSR desde: {model_path}")
            
            # Importar arquitectura EDSR
            try:
                from basicsr.archs.edsr_arch import EDSR
            except ImportError:
                logger.error("❌ No se pudo importar EDSR de basicsr. Instala basicsr: pip install basicsr")
                return None
            
            scale = config.get("scale", 2)
            
            # Configuración estándar de EDSR
            model = EDSR(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=256,
                num_block=32,
                upscale=scale,
                res_scale=0.1,
                img_range=255.,
                rgb_mean=[0.5, 0.5, 0.5]
            )
            
            # Cargar checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Manejar diferentes formatos de checkpoint
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
                logger.info("Usando parámetros EMA")
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
                logger.info("Usando parámetros regulares")
            else:
                state_dict = checkpoint
                logger.info("Usando state_dict directamente")
            
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model = model.to(self.device)
            
            logger.info("✅ EDSR cargado correctamente")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error cargando EDSR: {e}")
            return None
    
    def load_model(self, model_name: str) -> bool:
        """Carga un modelo específico por nombre"""
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Modelo desconocido: {model_name}")
            return False
        
        if model_name in self.loaded_models:
            logger.info(f"Modelo {model_name} ya está cargado")
            return True
        
        config = MODEL_CONFIGS[model_name]
        model_path = config["path"]
        
        if not os.path.exists(model_path):
            logger.error(f"Archivo no encontrado: {model_path}")
            return False
        
        # Cargar según tipo
        model = None
        if config["type"] == "tensorflow":
            model = self.load_esrgan_model(model_path)
        elif config["type"] == "pytorch_swinir":
            model = self.load_swinir_model(model_path, config)
        elif config["type"] == "pytorch_edsr":
            model = self.load_edsr_model(model_path, config)
        
        if model is not None:
            self.loaded_models[model_name] = {
                "model": model,
                "config": config,
                "type": config["type"]
            }
            logger.info(f"✅ Modelo {model_name} cargado exitosamente")
            return True
        else:
            logger.error(f"❌ Error cargando modelo {model_name}")
            return False
    
    def load_all_available_models(self) -> Dict[str, bool]:
        """Carga todos los modelos disponibles"""
        results = {}
        
        logger.info("🔄 Cargando todos los modelos disponibles...")
        
        for model_name, config in MODEL_CONFIGS.items():
            if os.path.exists(config["path"]):
                success = self.load_model(model_name)
                results[model_name] = success
            else:
                logger.warning(f"⚠️  Modelo no encontrado: {model_name}")
                results[model_name] = False
        
        loaded_count = sum(results.values())
        total_count = len(results)
        
        logger.info(f"📊 Modelos cargados: {loaded_count}/{total_count}")
        
        return results
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene un modelo cargado"""
        return self.loaded_models.get(model_name)
    
    def is_loaded(self, model_name: str) -> bool:
        """Verifica si un modelo está cargado"""
        return model_name in self.loaded_models
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene todos los modelos cargados"""
        return self.loaded_models
    
    def unload_model(self, model_name: str) -> bool:
        """Descarga un modelo de la memoria"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            # Limpiar cache de GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            logger.info(f"🗑️  Modelo {model_name} descargado")
            return True
        return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Obtiene información del uso de memoria"""
        info = {
            "loaded_models": len(self.loaded_models),
            "device": str(self.device)
        }
        
        if self.device.type == 'cuda':
            info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(self.device) / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved(self.device) / 1024**3,
                "gpu_memory_free": torch.cuda.mem_get_info(self.device)[0] / 1024**3,
                "gpu_memory_total": torch.cuda.mem_get_info(self.device)[1] / 1024**3
            })
        
        return info

# Instancia global del cargador
model_loader = ModelLoader()