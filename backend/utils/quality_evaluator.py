# backend/utils/quality_evaluator.py - Versión mejorada

"""
Evaluador de calidad de imágenes mejorado para superresolución
Incluye KimiaNet, MSE, y evaluación cualitativa
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
import os
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class KimiaNetPerceptualLoss:
    """Índice perceptual usando DenseNet121 con pesos KimiaNet para histopatología"""
    
    def __init__(self, kimianet_weights_path: str = None):
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
            try:
                feature_layer = self.densenet.get_layer('conv4_block6_concat')
            except:
                try:
                    feature_layer = self.densenet.get_layer('conv4_block24_concat')
                except:
                    feature_layer = self.densenet.layers[-2]
            
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
        """Calcula distancia perceptual entre dos imágenes usando KimiaNet"""
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


class QualitativeEvaluator:
    """Evaluador cualitativo para análisis visual de diferencias"""
    
    @staticmethod
    def calculate_difference_map(img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """Calcula mapa de diferencias entre dos imágenes"""
        try:
            # Asegurar mismo tamaño
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Convertir a float para cálculos precisos
            img1_float = img1.astype(np.float32)
            img2_float = img2.astype(np.float32)
            
            # Calcular diferencia absoluta
            diff_abs = np.abs(img1_float - img2_float)
            
            # Normalizar a [0, 255]
            diff_normalized = (diff_abs / np.max(diff_abs) * 255).astype(np.uint8)
            
            # Crear mapa de calor
            diff_heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
            
            # Estadísticas
            mean_diff = np.mean(diff_abs)
            max_diff = np.max(diff_abs)
            min_diff = np.min(diff_abs)
            std_diff = np.std(diff_abs)
            
            return {
                "difference_map": diff_normalized,
                "difference_heatmap": diff_heatmap,
                "statistics": {
                    "mean_difference": float(mean_diff),
                    "max_difference": float(max_diff),
                    "min_difference": float(min_diff),
                    "std_difference": float(std_diff),
                    "uniformity": float(std_diff / mean_diff) if mean_diff > 0 else 0.0
                },
                "analysis_success": True
            }
            
        except Exception as e:
            logger.error(f"Error en análisis cualitativo: {e}")
            return {
                "analysis_success": False,
                "error": str(e)
            }
    
    @staticmethod
    def generate_comparison_plot(original: np.ndarray, enhanced: np.ndarray, 
                               diff_map: np.ndarray, title: str = "Análisis Cualitativo") -> str:
        """Genera plot comparativo y lo retorna como base64"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Imagen original
            axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original (LR)', fontsize=12)
            axes[0, 0].axis('off')
            
            # Imagen mejorada
            axes[0, 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Mejorada (SR)', fontsize=12)
            axes[0, 1].axis('off')
            
            # Mapa de diferencias
            axes[1, 0].imshow(diff_map, cmap='gray')
            axes[1, 0].set_title('Mapa de Diferencias', fontsize=12)
            axes[1, 0].axis('off')
            
            # Histograma de diferencias
            axes[1, 1].hist(diff_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 1].set_title('Histograma de Diferencias', fontsize=12)
            axes[1, 1].set_xlabel('Intensidad de Diferencia')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convertir a base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_base64
            
        except Exception as e:
            logger.error(f"Error generando plot comparativo: {e}")
            return ""


class QualityEvaluator:
    """Evaluador de calidad mejorado con todas las métricas"""
    
    def __init__(self, kimianet_weights_path: str = None):
        self.kimianet_evaluator = None
        self.qualitative_evaluator = QualitativeEvaluator()
        self.kimianet_weights_path = kimianet_weights_path
        
        # Inicializar KimiaNet
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
    
    def calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula Mean Squared Error"""
        try:
            img1_float = img1.astype(np.float32)
            img2_float = img2.astype(np.float32)
            mse = np.mean((img1_float - img2_float) ** 2)
            return float(mse)
        except Exception as e:
            logger.error(f"Error calculando MSE: {e}")
            return -1.0
    
    
    def calculate_ms_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula MS-SSIM entre dos imágenes (implementación simplificada)"""
        try:
            # Convertir a escala de grises si es necesario
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1
                img2_gray = img2
            
            # Implementación simplificada de MS-SSIM usando múltiples escalas
            scales = [1.0, 0.5, 0.25]  # Tres escalas
            ms_ssim_values = []
            
            for scale in scales:
                if scale < 1.0:
                    new_height = int(img1_gray.shape[0] * scale)
                    new_width = int(img1_gray.shape[1] * scale)
                    
                    img1_scaled = cv2.resize(img1_gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    img2_scaled = cv2.resize(img2_gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                else:
                    img1_scaled = img1_gray
                    img2_scaled = img2_gray
                
                # Calcular SSIM en esta escala
                ssim_val = self.calculate_ssim(
                    np.expand_dims(img1_scaled, axis=2) if img1_scaled.ndim == 2 else img1_scaled,
                    np.expand_dims(img2_scaled, axis=2) if img2_scaled.ndim == 2 else img2_scaled
                )
                ms_ssim_values.append(ssim_val)
            
            # Promedio ponderado (dar más peso a escalas más altas)
            weights = [0.5, 0.3, 0.2]
            ms_ssim = sum(w * val for w, val in zip(weights, ms_ssim_values))
            
            return float(ms_ssim)
            
        except Exception as e:
            logger.error(f"Error calculando MS-SSIM: {e}")
            return -1.0
    
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula PSNR mejorado"""
        try:
            mse = self.calculate_mse(img1, img2)
            if mse == 0:
                return float('inf')
            
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return float(psnr)
        except Exception as e:
            logger.error(f"Error calculando PSNR: {e}")
            return 0.0
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula SSIM mejorado"""
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
    
    
    
    def calculate_fid_simplified(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula una versión simplificada de FID usando características básicas"""
        try:
            # Convertir a escala de grises
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1
                img2_gray = img2
            
            # Calcular estadísticas básicas como proxy de características
            def get_image_stats(img):
                img_float = img.astype(np.float32) / 255.0
                
                # Estadísticas de primer orden
                mean = np.mean(img_float)
                std = np.std(img_float)
                
                # Gradientes para capturar textura
                grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                grad_mean = np.mean(grad_mag)
                grad_std = np.std(grad_mag)
                
                # Características de textura usando filtros Gabor simplificados
                gabor_responses = []
                for angle in [0, 45, 90, 135]:
                    kernel = cv2.getGaborKernel((15, 15), 3, np.radians(angle), 2*np.pi/4, 0.5, 0, ktype=cv2.CV_32F)
                    response = cv2.filter2D(img_float, cv2.CV_32F, kernel)
                    gabor_responses.extend([np.mean(response), np.std(response)])
                
                return np.array([mean, std, grad_mean, grad_std] + gabor_responses)
            
            # Obtener características
            features1 = get_image_stats(img1_gray)
            features2 = get_image_stats(img2_gray)
            
            # Calcular distancia euclidiana como proxy de FID
            fid_score = np.linalg.norm(features1 - features2)
            
            return float(fid_score)
            
        except Exception as e:
            logger.error(f"Error calculando FID simplificado: {e}")
            return -1.0
    
    
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
    
    def evaluate_image_quality_comprehensive(self, 
                                           original: np.ndarray, 
                                           enhanced: np.ndarray,
                                           calculate_perceptual: bool = True,
                                           calculate_qualitative: bool = True) -> Dict[str, Any]:
        """
        Evaluación completa de calidad de imagen
        """
        try:
            # Asegurar que las imágenes tengan las mismas dimensiones
            original, enhanced = self.resize_to_match(original, enhanced)
            
            results = {
                "evaluation_success": False,
                "metrics": {},
                "qualitative_analysis": {},
                "error_message": None
            }
            
            # Métricas básicas
            logger.info("Calculando métricas básicas...")
            
            mse = self.calculate_mse(original, enhanced)
            psnr = self.calculate_psnr(original, enhanced)
            ssim = self.calculate_ssim(original, enhanced)
            
            results["metrics"] = {
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
                "perceptual_index": -1.0
            }
            
            # Índice perceptual con KimiaNet
            if calculate_perceptual:
                logger.info("Calculando índice perceptual con KimiaNet...")
                perceptual_index = self.calculate_perceptual_index(original, enhanced)
                results["metrics"]["perceptual_index"] = perceptual_index
            
            # Análisis cualitativo
            if calculate_qualitative:
                logger.info("Realizando análisis cualitativo...")
                qualitative_results = self.qualitative_evaluator.calculate_difference_map(
                    original, enhanced
                )
                
                if qualitative_results["analysis_success"]:
                    # Generar plot comparativo
                    comparison_plot = self.qualitative_evaluator.generate_comparison_plot(
                        original, enhanced, qualitative_results["difference_map"],
                        "Análisis Cualitativo - Superresolución"
                    )
                    
                    results["qualitative_analysis"] = {
                        "difference_statistics": qualitative_results["statistics"],
                        "difference_map": base64.b64encode(
                            cv2.imencode('.png', qualitative_results["difference_map"])[1]
                        ).decode(),
                        "difference_heatmap": base64.b64encode(
                            cv2.imencode('.png', qualitative_results["difference_heatmap"])[1]
                        ).decode(),
                        "comparison_plot": comparison_plot
                    }
            
            results["evaluation_success"] = True
            
            logger.info(f"Evaluación completada - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, MSE: {mse:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en evaluación completa de calidad: {e}")
            return {
                "evaluation_success": False,
                "metrics": {},
                "qualitative_analysis": {},
                "error_message": str(e)
            }
    
    
    
    def evaluate_comprehensive_quality(self, 
                                    original: np.ndarray, 
                                    enhanced: np.ndarray,
                                    calculate_perceptual: bool = True,
                                    calculate_advanced: bool = True) -> Dict[str, Any]:
        """
        Evaluación comprehensiva de calidad con todas las métricas disponibles
        """
        try:
            # Asegurar que las imágenes tengan las mismas dimensiones
            original, enhanced = self.resize_to_match(original, enhanced)
            
            results = {
                "evaluation_success": False,
                "error_message": None,
                "basic_metrics": {},
                "advanced_metrics": {},
                "perceptual_metrics": {},
                "image_info": {}
            }
            
            # Información de las imágenes
            results["image_info"] = {
                "original_size": f"{original.shape[1]}x{original.shape[0]}",
                "enhanced_size": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
                "channels": original.shape[2] if len(original.shape) == 3 else 1,
                "dtype": str(original.dtype)
            }
            
            # Métricas básicas
            logger.info("Calculando métricas básicas...")
            results["basic_metrics"] = {
                "psnr": self.calculate_psnr(original, enhanced),
                "ssim": self.calculate_ssim(original, enhanced),
                "mse": self.calculate_mse(original, enhanced)
            }
            
            # Métricas avanzadas
            if calculate_advanced:
                logger.info("Calculando métricas avanzadas...")
                results["advanced_metrics"] = {
                    "ms_ssim": self.calculate_ms_ssim(original, enhanced),
                    "fid_simplified": self.calculate_fid_simplified(original, enhanced)
                }
            
            # Métricas perceptuales
            if calculate_perceptual:
                logger.info("Calculando métricas perceptuales...")
                perceptual_index = self.calculate_perceptual_index(original, enhanced)
                results["perceptual_metrics"] = {
                    "perceptual_index": perceptual_index,
                    "kimianet_available": self.is_kimianet_available()
                }
            
            results["evaluation_success"] = True
            
            logger.info("Evaluación comprehensiva completada")
            return results
            
        except Exception as e:
            logger.error(f"Error en evaluación comprehensiva: {e}")
            results["error_message"] = str(e)
            return results
    
    
    
    def get_comprehensive_interpretation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Proporciona interpretación comprehensiva de todas las métricas"""
        interpretation = {
            "basic_metrics": {},
            "advanced_metrics": {},
            "perceptual_metrics": {},
            "overall_assessment": {}
        }
        
        # Interpretación de métricas básicas
        basic = results.get("basic_metrics", {})
        
        psnr = basic.get("psnr", -1)
        if psnr > 35:
            interpretation["basic_metrics"]["psnr"] = "Excelente (>35 dB)"
            psnr_score = 5
        elif psnr > 30:
            interpretation["basic_metrics"]["psnr"] = "Muy buena (30-35 dB)"
            psnr_score = 4
        elif psnr > 25:
            interpretation["basic_metrics"]["psnr"] = "Buena (25-30 dB)"
            psnr_score = 3
        elif psnr > 20:
            interpretation["basic_metrics"]["psnr"] = "Aceptable (20-25 dB)"
            psnr_score = 2
        elif psnr > 0:
            interpretation["basic_metrics"]["psnr"] = "Baja (<20 dB)"
            psnr_score = 1
        else:
            interpretation["basic_metrics"]["psnr"] = "No disponible"
            psnr_score = 0
        
        ssim = basic.get("ssim", -1)
        if ssim > 0.95:
            interpretation["basic_metrics"]["ssim"] = "Excelente (>0.95)"
            ssim_score = 5
        elif ssim > 0.9:
            interpretation["basic_metrics"]["ssim"] = "Muy buena (0.9-0.95)"
            ssim_score = 4
        elif ssim > 0.8:
            interpretation["basic_metrics"]["ssim"] = "Buena (0.8-0.9)"
            ssim_score = 3
        elif ssim > 0.7:
            interpretation["basic_metrics"]["ssim"] = "Aceptable (0.7-0.8)"
            ssim_score = 2
        elif ssim > 0:
            interpretation["basic_metrics"]["ssim"] = "Baja (<0.7)"
            ssim_score = 1
        else:
            interpretation["basic_metrics"]["ssim"] = "No disponible"
            ssim_score = 0
        
        mse = basic.get("mse", -1)
        if 0 <= mse < 100:
            interpretation["basic_metrics"]["mse"] = "Excelente (<100)"
        elif mse < 400:
            interpretation["basic_metrics"]["mse"] = "Buena (100-400)"
        elif mse < 1000:
            interpretation["basic_metrics"]["mse"] = "Aceptable (400-1000)"
        elif mse >= 1000:
            interpretation["basic_metrics"]["mse"] = "Alta (>1000)"
        else:
            interpretation["basic_metrics"]["mse"] = "No disponible"
        
        # Interpretación de métricas avanzadas
        advanced = results.get("advanced_metrics", {})
        
        ms_ssim = advanced.get("ms_ssim", -1)
        if ms_ssim > 0.95:
            interpretation["advanced_metrics"]["ms_ssim"] = "Excelente (>0.95)"
        elif ms_ssim > 0.9:
            interpretation["advanced_metrics"]["ms_ssim"] = "Muy buena (0.9-0.95)"
        elif ms_ssim > 0.8:
            interpretation["advanced_metrics"]["ms_ssim"] = "Buena (0.8-0.9)"
        elif ms_ssim > 0.7:
            interpretation["advanced_metrics"]["ms_ssim"] = "Aceptable (0.7-0.8)"
        elif ms_ssim > 0:
            interpretation["advanced_metrics"]["ms_ssim"] = "Baja (<0.7)"
        else:
            interpretation["advanced_metrics"]["ms_ssim"] = "No disponible"
        
        fid = advanced.get("fid_simplified", -1)
        if 0 <= fid < 1:
            interpretation["advanced_metrics"]["fid"] = "Excelente (<1.0)"
        elif fid < 3:
            interpretation["advanced_metrics"]["fid"] = "Buena (1.0-3.0)"
        elif fid < 5:
            interpretation["advanced_metrics"]["fid"] = "Aceptable (3.0-5.0)"
        elif fid >= 5:
            interpretation["advanced_metrics"]["fid"] = "Alta (>5.0)"
        else:
            interpretation["advanced_metrics"]["fid"] = "No disponible"
        
        # Interpretación perceptual
        perceptual = results.get("perceptual_metrics", {})
        perceptual_idx = perceptual.get("perceptual_index", -1)
        
        if perceptual_idx < 0.001:
            interpretation["perceptual_metrics"]["perceptual"] = "Excelente (<0.001)"
            perceptual_score = 5
        elif perceptual_idx < 0.01:
            interpretation["perceptual_metrics"]["perceptual"] = "Muy buena (0.001-0.01)"
            perceptual_score = 4
        elif perceptual_idx < 0.05:
            interpretation["perceptual_metrics"]["perceptual"] = "Buena (0.01-0.05)"
            perceptual_score = 3
        elif perceptual_idx < 0.1:
            interpretation["perceptual_metrics"]["perceptual"] = "Aceptable (0.05-0.1)"
            perceptual_score = 2
        elif perceptual_idx >= 0.1:
            interpretation["perceptual_metrics"]["perceptual"] = "Baja (>0.1)"
            perceptual_score = 1
        else:
            interpretation["perceptual_metrics"]["perceptual"] = "No disponible"
            perceptual_score = 0
        
        # Evaluación general
        if psnr_score > 0 and ssim_score > 0:
            overall_score = (psnr_score + ssim_score + perceptual_score) / 3
            
            if overall_score >= 4.5:
                interpretation["overall_assessment"] = {
                    "quality": "Excelente",
                    "score": overall_score,
                    "recommendation": "La imagen procesada tiene calidad superior"
                }
            elif overall_score >= 3.5:
                interpretation["overall_assessment"] = {
                    "quality": "Muy buena",
                    "score": overall_score,
                    "recommendation": "La imagen procesada tiene muy buena calidad"
                }
            elif overall_score >= 2.5:
                interpretation["overall_assessment"] = {
                    "quality": "Buena",
                    "score": overall_score,
                    "recommendation": "La imagen procesada tiene calidad aceptable"
                }
            elif overall_score >= 1.5:
                interpretation["overall_assessment"] = {
                    "quality": "Regular",
                    "score": overall_score,
                    "recommendation": "La imagen procesada necesita mejoras"
                }
            else:
                interpretation["overall_assessment"] = {
                    "quality": "Baja",
                    "score": overall_score,
                    "recommendation": "La imagen procesada tiene calidad baja"
                }
        else:
            interpretation["overall_assessment"] = {
                "quality": "No evaluable",
                "score": 0,
                "recommendation": "No se pudieron calcular métricas suficientes"
            }
        
        return interpretation
    
    
    
    def get_quality_interpretation_comprehensive(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Interpretación completa de métricas de calidad"""
        interpretation = {}
        metrics = results.get("metrics", {})
        
        # PSNR
        psnr = metrics.get("psnr", -1)
        if psnr > 35:
            interpretation["psnr"] = "Excelente (>35 dB) - Calidad muy alta"
        elif psnr > 30:
            interpretation["psnr"] = "Muy buena (30-35 dB) - Calidad alta"
        elif psnr > 25:
            interpretation["psnr"] = "Buena (25-30 dB) - Calidad aceptable"
        elif psnr > 20:
            interpretation["psnr"] = "Regular (20-25 dB) - Calidad baja"
        elif psnr > 0:
            interpretation["psnr"] = "Mala (<20 dB) - Calidad muy baja"
        else:
            interpretation["psnr"] = "No disponible"
        
        # SSIM
        ssim = metrics.get("ssim", -1)
        if ssim > 0.95:
            interpretation["ssim"] = "Excelente (>0.95) - Muy similar estructuralmente"
        elif ssim > 0.9:
            interpretation["ssim"] = "Muy buena (0.9-0.95) - Bastante similar"
        elif ssim > 0.8:
            interpretation["ssim"] = "Buena (0.8-0.9) - Similaridad aceptable"
        elif ssim > 0.7:
            interpretation["ssim"] = "Regular (0.7-0.8) - Similaridad limitada"
        elif ssim > 0:
            interpretation["ssim"] = "Mala (<0.7) - Poca similaridad"
        else:
            interpretation["ssim"] = "No disponible"
        
        # MSE
        mse = metrics.get("mse", -1)
        if mse < 100:
            interpretation["mse"] = "Excelente (<100) - Error muy bajo"
        elif mse < 500:
            interpretation["mse"] = "Buena (100-500) - Error bajo"
        elif mse < 1000:
            interpretation["mse"] = "Regular (500-1000) - Error moderado"
        elif mse < 2000:
            interpretation["mse"] = "Mala (1000-2000) - Error alto"
        elif mse >= 2000:
            interpretation["mse"] = "Muy mala (>2000) - Error muy alto"
        else:
            interpretation["mse"] = "No disponible"
        
        # Índice Perceptual
        perceptual = metrics.get("perceptual_index", -1)
        if perceptual < 0.001:
            interpretation["perceptual"] = "Excelente (<0.001) - Muy similar perceptualmente"
        elif perceptual < 0.01:
            interpretation["perceptual"] = "Buena (0.001-0.01) - Bastante similar"
        elif perceptual < 0.1:
            interpretation["perceptual"] = "Aceptable (0.01-0.1) - Similaridad moderada"
        elif perceptual >= 0.1:
            interpretation["perceptual"] = "Baja (>0.1) - Poca similaridad perceptual"
        else:
            interpretation["perceptual"] = "No disponible (KimiaNet no cargado)"
        
        # Análisis cualitativo
        qualitative = results.get("qualitative_analysis", {})
        if "difference_statistics" in qualitative:
            stats = qualitative["difference_statistics"]
            uniformity = stats.get("uniformity", -1)
            
            if uniformity < 0.5:
                interpretation["qualitative"] = "Diferencias uniformes - Mejora consistente"
            elif uniformity < 1.0:
                interpretation["qualitative"] = "Diferencias moderadas - Mejora variable"
            else:
                interpretation["qualitative"] = "Diferencias irregulares - Mejora inconsistente"
        
        return interpretation
    
    def is_kimianet_available(self) -> bool:
        """Verifica si KimiaNet está disponible"""
        return (self.kimianet_evaluator is not None and 
                self.kimianet_evaluator.is_loaded)

# Instancia global del evaluador mejorado
quality_evaluator = QualityEvaluator()