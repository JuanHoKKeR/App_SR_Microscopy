"""
Evaluador de calidad de imágenes usando KimiaNet y métricas tradicionales
Basado en el script evaluate_model.py del usuario
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
import os
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class KimiaNetPerceptualLoss:
    """Índice perceptual usando DenseNet121 con pesos KimiaNet para histopatología"""
    
    def __init__(self, kimianet_weights_path: str = None):
        """
        Inicializa el evaluador perceptual con KimiaNet
        
        Args:
            kimianet_weights_path: Ruta a los pesos KimiaNet (.h5)
        """
        self.kimianet_weights_path = kimianet_weights_path or "../models/model-kimianet/KimiaNetKerasWeights.h5"
        self.feature_extractor = None
        self.is_loaded = False
        
        self._load_kimianet()
    
    def _load_kimianet(self):
        """Carga DenseNet121 con pesos KimiaNet"""
        try:
            logger.info("🧠 Cargando DenseNet121 con pesos KimiaNet...")
            
            # Cargar DenseNet121 sin la capa final
            self.densenet = DenseNet121(
                include_top=False, 
                weights=None,  # Sin pesos de ImageNet
                input_shape=(None, None, 3)
            )
            
            # Cargar pesos KimiaNet si existe el archivo
            if os.path.exists(self.kimianet_weights_path):
                try:
                    self.densenet.load_weights(self.kimianet_weights_path)
                    logger.info(f"✅ Pesos KimiaNet cargados desde: {self.kimianet_weights_path}")
                except Exception as e:
                    logger.warning(f"⚠️  Error cargando pesos KimiaNet: {e}")
                    logger.warning("    Usando DenseNet121 sin preentrenar")
            else:
                logger.warning(f"⚠️  No se encontraron pesos KimiaNet en: {self.kimianet_weights_path}")
                logger.warning("    Usando DenseNet121 sin preentrenar")
            
            # Usar una capa intermedia para extraer características
            # conv4_block6_concat es una buena capa para características semánticas
            try:
                feature_layer = self.densenet.get_layer('conv4_block6_concat')
            except:
                # Si no existe esa capa, usar una alternativa
                try:
                    feature_layer = self.densenet.get_layer('conv4_block24_concat')
                except:
                    # Como último recurso, usar la salida completa
                    feature_layer = self.densenet.layers[-2]  # Antes del GlobalAveragePooling
            
            self.feature_extractor = tf.keras.Model(
                inputs=self.densenet.input,
                outputs=feature_layer.output
            )
            
            # Congelar el modelo  
            for layer in self.feature_extractor.layers:
                layer.trainable = False
                
            self.is_loaded = True
            logger.info(f"✅ Extractor de características listo: {feature_layer.name}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando KimiaNet: {e}")
            self.is_loaded = False
    
    def calculate_perceptual_distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calcula distancia perceptual entre dos imágenes usando KimiaNet
        
        Args:
            img1, img2: Imágenes en formato [H, W, 3] con valores [0, 255]
            
        Returns:
            Distancia perceptual (más bajo = más similar)
        """
        if not self.is_loaded:
            logger.error("KimiaNet no está cargado")
            return -1.0
        
        try:
            # Convertir a tensores TensorFlow
            img1_tf = tf.convert_to_tensor(img1, dtype=tf.float32)
            img2_tf = tf.convert_to_tensor(img2, dtype=tf.float32)
            
            # Agregar dimensión de batch si no existe
            if len(img1_tf.shape) == 3:
                img1_tf = tf.expand_dims(img1_tf, 0)
            if len(img2_tf.shape) == 3:
                img2_tf = tf.expand_dims(img2_tf, 0)
            
            # Normalizar para DenseNet (adaptado para histopatología)
            img1_norm = (img1_tf - 127.5) / 127.5  # [-1, 1]
            img2_norm = (img2_tf - 127.5) / 127.5  # [-1, 1]
            
            # Extraer características
            features1 = self.feature_extractor(img1_norm)
            features2 = self.feature_extractor(img2_norm)
            
            # Calcular distancia L2 entre características
            perceptual_distance = tf.reduce_mean(tf.square(features1 - features2))
            
            return float(perceptual_distance.numpy())
            
        except Exception as e:
            logger.error(f"Error calculando distancia perceptual: {e}")
            return -1.0


class QualityEvaluator:
    """Evaluador de calidad de imágenes para superresolución"""
    
    def __init__(self, kimianet_weights_path: str = None):
        """
        Inicializa el evaluador de calidad
        
        Args:
            kimianet_weights_path: Ruta a los pesos KimiaNet
        """
        self.kimianet_evaluator = None
        self.kimianet_weights_path = kimianet_weights_path
        
        # Intentar cargar KimiaNet
        self._initialize_kimianet()
    
    def _initialize_kimianet(self):
        """Inicializa KimiaNet de forma lazy"""
        try:
            self.kimianet_evaluator = KimiaNetPerceptualLoss(self.kimianet_weights_path)
            if self.kimianet_evaluator.is_loaded:
                logger.info("✅ KimiaNet evaluator inicializado correctamente")
            else:
                logger.warning("⚠️  KimiaNet evaluator no se pudo cargar completamente")
        except Exception as e:
            logger.error(f"Error inicializando KimiaNet: {e}")
            self.kimianet_evaluator = None
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula PSNR entre dos imágenes"""
        try:
            # Convertir a float32 para cálculo preciso
            img1_float = img1.astype(np.float32)
            img2_float = img2.astype(np.float32)
            
            # Calcular MSE
            mse = np.mean((img1_float - img2_float) ** 2)
            
            if mse == 0:
                return float('inf')
            
            # Calcular PSNR
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            return float(psnr)
            
        except Exception as e:
            logger.error(f"Error calculando PSNR: {e}")
            return 0.0
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula SSIM entre dos imágenes"""
        try:
            # Convertir a escala de grises si es necesario
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1
                img2_gray = img2
            
            # Convertir a float
            img1_float = img1_gray.astype(np.float32)
            img2_float = img2_gray.astype(np.float32)
            
            # Parámetros SSIM
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            # Calcular medias con filtro gaussiano
            mu1 = cv2.GaussianBlur(img1_float, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2_float, (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            # Calcular varianzas y covarianza
            sigma1_sq = cv2.GaussianBlur(img1_float ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2_float ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(img1_float * img2_float, (11, 11), 1.5) - mu1_mu2
            
            # Calcular SSIM
            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            
            ssim_map = numerator / denominator
            ssim = np.mean(ssim_map)
            
            return float(ssim)
            
        except Exception as e:
            logger.error(f"Error calculando SSIM: {e}")
            return 0.0
    
    def calculate_perceptual_index(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula índice perceptual usando KimiaNet"""
        if self.kimianet_evaluator is None or not self.kimianet_evaluator.is_loaded:
            logger.warning("KimiaNet no disponible para cálculo perceptual")
            return -1.0
        
        return self.kimianet_evaluator.calculate_perceptual_distance(img1, img2)
    
    def resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Redimensiona imágenes para que tengan las mismas dimensiones"""
        if img1.shape != img2.shape:
            logger.info(f"Redimensionando imagen: {img2.shape} → {img1.shape}")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        return img1, img2
    
    def evaluate_image_quality(self, 
                             original: np.ndarray, 
                             enhanced: np.ndarray,
                             calculate_perceptual: bool = True) -> Dict[str, Any]:
        """
        Evalúa la calidad de una imagen procesada vs original
        
        Args:
            original: Imagen original (referencia)
            enhanced: Imagen procesada
            calculate_perceptual: Si calcular índice perceptual (más lento)
            
        Returns:
            Dict con métricas de calidad
        """
        try:
            # Asegurar que las imágenes tengan las mismas dimensiones
            original, enhanced = self.resize_to_match(original, enhanced)
            
            results = {
                "psnr": -1.0,
                "ssim": -1.0,
                "perceptual_index": -1.0,
                "evaluation_success": False,
                "error_message": None
            }
            
            # Calcular PSNR
            psnr = self.calculate_psnr(original, enhanced)
            results["psnr"] = psnr
            
            # Calcular SSIM
            ssim = self.calculate_ssim(original, enhanced)
            results["ssim"] = ssim
            
            # Calcular índice perceptual si se solicita
            if calculate_perceptual:
                perceptual_index = self.calculate_perceptual_index(original, enhanced)
                results["perceptual_index"] = perceptual_index
            
            results["evaluation_success"] = True
            
            logger.info(f"Evaluación completada - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en evaluación de calidad: {e}")
            return {
                "psnr": -1.0,
                "ssim": -1.0,
                "perceptual_index": -1.0,
                "evaluation_success": False,
                "error_message": str(e)
            }
    
    def get_quality_interpretation(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Proporciona interpretación de las métricas de calidad"""
        interpretation = {}
        
        # PSNR
        psnr = results.get("psnr", -1)
        if psnr > 30:
            interpretation["psnr"] = "Excelente (>30 dB)"
        elif psnr > 25:
            interpretation["psnr"] = "Buena (25-30 dB)"
        elif psnr > 20:
            interpretation["psnr"] = "Aceptable (20-25 dB)"
        elif psnr > 0:
            interpretation["psnr"] = "Baja (<20 dB)"
        else:
            interpretation["psnr"] = "No disponible"
        
        # SSIM
        ssim = results.get("ssim", -1)
        if ssim > 0.9:
            interpretation["ssim"] = "Excelente (>0.9)"
        elif ssim > 0.8:
            interpretation["ssim"] = "Buena (0.8-0.9)"
        elif ssim > 0.7:
            interpretation["ssim"] = "Aceptable (0.7-0.8)"
        elif ssim > 0:
            interpretation["ssim"] = "Baja (<0.7)"
        else:
            interpretation["ssim"] = "No disponible"
        
        # Índice Perceptual
        perceptual = results.get("perceptual_index", -1)
        if perceptual < 0.001:
            interpretation["perceptual"] = "Excelente (<0.001)"
        elif perceptual < 0.01:
            interpretation["perceptual"] = "Buena (0.001-0.01)"
        elif perceptual < 0.1:
            interpretation["perceptual"] = "Aceptable (0.01-0.1)"
        elif perceptual >= 0.1:
            interpretation["perceptual"] = "Baja (>0.1)"
        else:
            interpretation["perceptual"] = "No disponible"
        
        return interpretation
    
    def is_kimianet_available(self) -> bool:
        """Verifica si KimiaNet está disponible"""
        return (self.kimianet_evaluator is not None and 
                self.kimianet_evaluator.is_loaded)

# Instancia global del evaluador
quality_evaluator = QualityEvaluator()