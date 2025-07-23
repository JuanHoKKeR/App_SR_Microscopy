"""
Cliente API para comunicación con el backend FastAPI
"""

import requests
import logging
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

logger = logging.getLogger(__name__)

class APIClient:
    """Cliente para comunicación con la API del backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.timeout = 60  # timeout en segundos
    
    def check_connection(self) -> bool:
        """Verifica la conexión con la API"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error conectando con API: {e}")
            return False
    
    def get_api_info(self) -> Optional[Dict[str, Any]]:
        """Obtiene información general de la API"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error obteniendo info de API: {e}")
            return None
    
    def get_model_recommendations(self, architecture: str) -> Dict[str, Any]:
        """Obtiene recomendaciones de configuración para una arquitectura"""
        try:
            arch_models = self.get_models_by_architecture(architecture)
            
            if not arch_models:
                return {"available_sizes": [], "max_scale": 1, "recommended_start": 256}
            
            available_models = [m for m in arch_models["models"] if m["available"]]
            
            if not available_models:
                return {"available_sizes": [], "max_scale": 1, "recommended_start": 256}
            
            # Calcular tamaños disponibles y escalas máximas
            input_sizes = sorted(list(set([m["input_size"] for m in available_models])))
            
            # Calcular escala máxima posible
            max_scale = 1
            for start_size in input_sizes:
                current_scale = 1
                current_size = start_size
                
                while True:
                    next_size = current_size * 2
                    has_model = any(
                        m["input_size"] == current_size and m["output_size"] == next_size 
                        for m in available_models
                    )
                    
                    if has_model:
                        current_scale *= 2
                        current_size = next_size
                        max_scale = max(max_scale, current_scale)
                    else:
                        break
            
            # Tamaño recomendado (el más común o el medio)
            recommended_start = input_sizes[len(input_sizes) // 2] if input_sizes else 256
            
            return {
                "available_sizes": input_sizes,
                "max_scale": max_scale,
                "recommended_start": recommended_start,
                "total_models": len(available_models),
                "architecture": architecture
            }
        except Exception as e:
            logger.error(f"Error obteniendo recomendaciones de modelo: {e}")
            return {"available_sizes": [], "max_scale": 1, "recommended_start": 256}
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Obtiene lista de modelos disponibles"""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=15)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error HTTP {response.status_code}: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error obteniendo modelos: {e}")
            return []
    
    def get_models_by_architecture(self, architecture: str) -> Optional[Dict[str, Any]]:
        """Obtiene modelos filtrados por arquitectura"""
        try:
            response = self.session.get(
                f"{self.base_url}/models/{architecture}", 
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error obteniendo modelos por arquitectura: {e}")
            return None
    
    def get_upsampling_path(self, architecture: str, start_size: int, target_scale: int) -> Optional[Dict[str, Any]]:
        """Obtiene la ruta de upsampling para parámetros específicos"""
        try:
            response = self.session.get(
                f"{self.base_url}/upsampling_path/{architecture}/{start_size}/{target_scale}",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error obteniendo ruta: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error obteniendo ruta de upsampling: {e}")
            return None
    
    def process_patch(self, 
                     image_file, 
                     model_name: str, 
                     x: int = 0, 
                     y: int = 0, 
                     width: int = 256, 
                     height: int = 256) -> Optional[Dict[str, Any]]:
        """Procesa un parche específico"""
        try:
            # Preparar archivo
            if hasattr(image_file, 'seek'):
                image_file.seek(0)
            
            files = {"file": ("image.png", image_file, "image/png")}
            data = {
                "model_name": model_name,
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }
            
            response = self.session.post(
                f"{self.base_url}/process_patch",
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error procesando parche: HTTP {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error en process_patch: {e}")
            return None
    
    def process_sequential_upsampling(self,
                                    image_file,
                                    architecture: str,
                                    start_size: int,
                                    target_scale: int,
                                    x: int = 0,
                                    y: int = 0,
                                    width: int = 256,
                                    height: int = 256,
                                    evaluate_quality: bool = False) -> Optional[Dict[str, Any]]:
        """Procesa upsampling secuencial"""
        try:
            # Preparar archivo
            if hasattr(image_file, 'seek'):
                image_file.seek(0)
            
            files = {"file": ("image.png", image_file, "image/png")}
            data = {
                "architecture": architecture,
                "start_size": start_size,
                "target_scale": target_scale,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "evaluate_quality": evaluate_quality
            }
            
            response = self.session.post(
                f"{self.base_url}/process_sequential",
                files=files,
                data=data,
                timeout=self.timeout * 2  # Más tiempo para procesamiento secuencial
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error en upsampling secuencial: HTTP {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error en process_sequential_upsampling: {e}")
            return None
    
    def load_model(self, model_name: str) -> bool:
        """Carga un modelo específico"""
        try:
            response = self.session.post(
                f"{self.base_url}/load_model/{model_name}",
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error cargando modelo {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Descarga un modelo específico"""
        try:
            response = self.session.delete(
                f"{self.base_url}/unload_model/{model_name}",
                timeout=15
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error descargando modelo {model_name}: {e}")
            return False
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas del sistema"""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return None
    
    @st.cache_data(ttl=30)  # Cache por 30 segundos
    def get_cached_models(_self) -> List[Dict[str, Any]]:
        """Versión cacheada de get_available_models"""
        return _self.get_available_models()
    
    def validate_upsampling_feasibility(self, architecture: str, start_size: int, target_scale: int) -> tuple[bool, str]:
        """Valida si es posible realizar el upsampling solicitado"""
        path_info = self.get_upsampling_path(architecture, start_size, target_scale)
        
        if not path_info:
            return False, f"No se puede alcanzar x{target_scale} desde {start_size} con {architecture}"
        
        if not path_info.get("all_models_available", False):
            missing_models = []
            for detail in path_info.get("path_details", []):
                if not detail.get("available", False):
                    missing_models.append(detail["model_name"])
            
            return False, f"Modelos no disponibles: {', '.join(missing_models)}"
        
        return True, f"Ruta válida: {len(path_info['path'])} pasos"
    
    def get_kimianet_status(self) -> Optional[Dict[str, Any]]:
        """Obtiene el estado de KimiaNet"""
        try:
            response = self.session.get(f"{self.base_url}/kimianet_status", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error obteniendo estado de KimiaNet: {e}")
            return None
    
    def evaluate_image_quality(self, 
                             original_file, 
                             enhanced_file,
                             calculate_perceptual: bool = True) -> Optional[Dict[str, Any]]:
        """Evalúa la calidad de una imagen procesada vs original"""
        try:
            # Preparar archivos
            if hasattr(original_file, 'seek'):
                original_file.seek(0)
            if hasattr(enhanced_file, 'seek'):
                enhanced_file.seek(0)
            
            files = {
                "original_file": ("original.png", original_file, "image/png"),
                "enhanced_file": ("enhanced.png", enhanced_file, "image/png")
            }
            data = {
                "calculate_perceptual": calculate_perceptual
            }
            
            response = self.session.post(
                f"{self.base_url}/evaluate_quality",
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error evaluando calidad: HTTP {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error en evaluate_image_quality: {e}")
            return None