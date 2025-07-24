"""
Utilidades para manejo de imágenes
Conversiones, validaciones y operaciones básicas
"""

import numpy as np
import cv2
import base64
import io
from PIL import Image
from typing import Tuple, Optional, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ImageUtils:
    """Utilidades para manejo de imágenes"""
    
    @staticmethod
    def image_to_base64(image: np.ndarray, format: str = 'PNG') -> str:
        """Convierte imagen numpy a base64"""
        try:
            _, buffer = cv2.imencode(f'.{format.lower()}', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return img_base64
        except Exception as e:
            logger.error(f"Error convirtiendo imagen a base64: {e}")
            return ""
    
    @staticmethod
    def base64_to_image(base64_str: str) -> Optional[np.ndarray]:
        """Convierte base64 a imagen numpy"""
        try:
            # Remover prefijo data:image si existe
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            
            img_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Error convirtiendo base64 a imagen: {e}")
            return None
    
    @staticmethod
    def bytes_to_image(img_bytes: bytes) -> Optional[np.ndarray]:
        """Convierte bytes a imagen numpy"""
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Error convirtiendo bytes a imagen: {e}")
            return None
    
    @staticmethod
    def pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
        """Convierte imagen PIL a formato OpenCV"""
        try:
            # Convertir a RGB si es necesario
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # PIL usa RGB, OpenCV usa BGR
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            logger.error(f"Error convirtiendo PIL a OpenCV: {e}")
            return None
    
    @staticmethod
    def opencv_to_pil(opencv_image: np.ndarray) -> Optional[Image.Image]:
        """Convierte imagen OpenCV a formato PIL"""
        try:
            # OpenCV usa BGR, PIL usa RGB
            rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            return pil_image
        except Exception as e:
            logger.error(f"Error convirtiendo OpenCV a PIL: {e}")
            return None
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """Valida que la imagen sea válida"""
        if image is None:
            return False
        
        if len(image.shape) not in [2, 3]:
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False
        
        return True
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """Obtiene información de la imagen"""
        if not ImageUtils.validate_image(image):
            return {}
        
        info = {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2] if len(image.shape) == 3 else 1,
            "dtype": str(image.dtype),
            "size_mb": image.nbytes / (1024 * 1024)
        }
        
        return info
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
        """Redimensiona imagen manteniendo calidad"""
        try:
            width, height = target_size
            resized = cv2.resize(image, (width, height), interpolation=interpolation)
            return resized
        except Exception as e:
            logger.error(f"Error redimensionando imagen: {e}")
            return image
    
    @staticmethod
    def pad_to_multiple(image: np.ndarray, multiple: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Aplica padding para que las dimensiones sean múltiplos de un valor"""
        h, w = image.shape[:2]
        
        h_pad = (h // multiple + 1) * multiple - h if h % multiple != 0 else 0
        w_pad = (w // multiple + 1) * multiple - w if w % multiple != 0 else 0
        
        if h_pad > 0 or w_pad > 0:
            padded = cv2.copyMakeBorder(
                image, 0, h_pad, 0, w_pad, 
                cv2.BORDER_REFLECT_101
            )
            return padded, (h, w)
        
        return image, (h, w)
    
    @staticmethod
    def crop_to_original_size(image: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Recorta imagen a su tamaño original (remueve padding)"""
        h, w = original_size
        return image[:h, :w]
    
    @staticmethod
    def normalize_image(image: np.ndarray, target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """Normaliza imagen al rango especificado"""
        image_float = image.astype(np.float32)
        
        # Normalizar de [0, 255] al rango objetivo
        min_val, max_val = target_range
        normalized = (image_float / 255.0) * (max_val - min_val) + min_val
        
        return normalized
    
    @staticmethod
    def denormalize_image(image: np.ndarray, source_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """Desnormaliza imagen de rango especificado a [0, 255]"""
        min_val, max_val = source_range
        
        # Desnormalizar al rango [0, 255]
        denormalized = (image - min_val) / (max_val - min_val) * 255.0
        denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)
        
        return denormalized
    
    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula PSNR entre dos imágenes"""
        try:
            mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
            if mse == 0:
                return float('inf')
            
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return float(psnr)
        except Exception as e:
            logger.error(f"Error calculando PSNR: {e}")
            return 0.0
    
    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula SSIM entre dos imágenes (implementación básica)"""
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
            
            # Calcular medias
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
    
    @staticmethod
    def split_image_into_patches(image: np.ndarray, patch_size: int, 
                               overlap: int = 0) -> list:
        """Divide imagen en parches"""
        patches = []
        h, w = image.shape[:2]
        
        stride = patch_size - overlap
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append({
                    'patch': patch,
                    'coordinates': (x, y),
                    'size': (patch_size, patch_size)
                })
        
        return patches
    
    @staticmethod
    def reconstruct_from_patches(patches: list, original_shape: Tuple[int, int], 
                               patch_size: int, overlap: int = 0) -> np.ndarray:
        """Reconstruye imagen desde parches"""
        h, w = original_shape[:2]
        
        if len(original_shape) == 3:
            reconstructed = np.zeros((h * 2, w * 2, original_shape[2]), dtype=np.uint8)  # x2 scale
        else:
            reconstructed = np.zeros((h * 2, w * 2), dtype=np.uint8)
        
        weight_map = np.zeros((h * 2, w * 2), dtype=np.float32)
        
        for patch_info in patches:
            patch = patch_info['enhanced_patch']
            x, y = patch_info['coordinates']
            
            # Coordenadas en imagen reconstruida (x2 scale)
            x_recon = x * 2
            y_recon = y * 2
            patch_h, patch_w = patch.shape[:2]
            
            # Agregar parche ponderado
            reconstructed[y_recon:y_recon+patch_h, x_recon:x_recon+patch_w] += patch
            weight_map[y_recon:y_recon+patch_h, x_recon:x_recon+patch_w] += 1
        
        # Normalizar por pesos
        weight_map[weight_map == 0] = 1
        if len(original_shape) == 3:
            for c in range(original_shape[2]):
                reconstructed[:, :, c] = reconstructed[:, :, c] / weight_map
        else:
            reconstructed = reconstructed / weight_map
        
        return reconstructed.astype(np.uint8)

    @staticmethod
    def split_image_for_processing(image: np.ndarray, patch_size: int, 
                                overlap: int = 32) -> list:
        """Divide imagen en parches para procesamiento con overlap"""
        patches = []
        h, w = image.shape[:2]
        
        stride = patch_size - overlap
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Ajustar límites para no exceder imagen
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                y_start = max(0, y_end - patch_size)
                x_start = max(0, x_end - patch_size)
                
                patch = image[y_start:y_end, x_start:x_end]
                
                patches.append({
                    'patch': patch,
                    'original_coords': (x_start, y_start),
                    'target_coords': (x_start * 2, y_start * 2),  # Asumiendo x2 scale
                    'size': (patch_size, patch_size)
                })
        
        return patches

    @staticmethod
    def reconstruct_image_from_patches(patches: list, original_size: Tuple[int, int],
                                    scale_factor: int, overlap: int = 32) -> np.ndarray:
        """Reconstruye imagen desde parches procesados"""
        h_orig, w_orig = original_size
        h_new, w_new = h_orig * scale_factor, w_orig * scale_factor
        
        # Determinar número de canales
        sample_patch = patches[0]['enhanced_patch']
        channels = sample_patch.shape[2] if len(sample_patch.shape) == 3 else 1
        
        if channels > 1:
            reconstructed = np.zeros((h_new, w_new, channels), dtype=np.float32)
            weight_map = np.zeros((h_new, w_new), dtype=np.float32)
        else:
            reconstructed = np.zeros((h_new, w_new), dtype=np.float32)
            weight_map = np.zeros((h_new, w_new), dtype=np.float32)
        
        for patch_info in patches:
            patch = patch_info['enhanced_patch'].astype(np.float32)
            x_target, y_target = patch_info['target_coords']
            patch_h, patch_w = patch.shape[:2]
            
            # Crear máscara de peso con fade en los bordes para suavizar
            weight_patch = np.ones((patch_h, patch_w), dtype=np.float32)
            
            if overlap > 0:
                # Aplicar fade en los bordes
                fade_size = min(overlap // 2, patch_h // 4, patch_w // 4)
                for i in range(fade_size):
                    weight_val = (i + 1) / fade_size
                    weight_patch[i, :] *= weight_val  # Top
                    weight_patch[-i-1, :] *= weight_val  # Bottom
                    weight_patch[:, i] *= weight_val  # Left
                    weight_patch[:, -i-1] *= weight_val  # Right
            
            # Agregar patch ponderado
            y_end = min(y_target + patch_h, h_new)
            x_end = min(x_target + patch_w, w_new)
            
            if channels > 1:
                for c in range(channels):
                    reconstructed[y_target:y_end, x_target:x_end, c] += \
                        patch[:y_end-y_target, :x_end-x_target, c] * weight_patch[:y_end-y_target, :x_end-x_target]
            else:
                reconstructed[y_target:y_end, x_target:x_end] += \
                    patch[:y_end-y_target, :x_end-x_target] * weight_patch[:y_end-y_target, :x_end-x_target]
            
            weight_map[y_target:y_end, x_target:x_end] += weight_patch[:y_end-y_target, :x_end-x_target]
        
        # Normalizar por pesos
        weight_map[weight_map == 0] = 1
        if channels > 1:
            for c in range(channels):
                reconstructed[:, :, c] /= weight_map
        else:
            reconstructed /= weight_map
        
        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    @staticmethod
    def calculate_optimal_strategy(input_size: Tuple[int, int], target_scale: int,
                                architecture: str) -> Dict[str, Any]:
        """Calcula la estrategia óptima para procesar imagen completa"""
        h, w = input_size
        target_h, target_w = h * target_scale, w * target_scale
        
        # Tamaños de patch disponibles para la arquitectura
        available_sizes = [64, 128, 256, 512]
        
        strategies = []
        
        # Estrategia 1: Imagen completa (si es posible)
        if h <= 1024 and w <= 1024:
            strategies.append({
                "type": "full_image",
                "description": f"Procesar imagen completa ({h}x{w})",
                "patch_count": 1,
                "overlap": 0,
                "memory_efficient": True
            })
        
        # Estrategia 2: División en parches
        for patch_size in available_sizes:
            if patch_size <= min(h, w):
                # Calcular número de parches necesarios
                overlap = min(64, patch_size // 4)
                stride = patch_size - overlap
                
                patches_h = max(1, (h - overlap + stride - 1) // stride)
                patches_w = max(1, (w - overlap + stride - 1) // stride)
                total_patches = patches_h * patches_w
                
                strategies.append({
                    "type": "patch_based",
                    "patch_size": patch_size,
                    "overlap": overlap,
                    "patches_h": patches_h,
                    "patches_w": patches_w,
                    "patch_count": total_patches,
                    "description": f"Dividir en {total_patches} parches de {patch_size}x{patch_size}",
                    "memory_efficient": total_patches <= 16
                })
        
        # Ordenar por eficiencia (menos parches = mejor)
        strategies.sort(key=lambda x: x["patch_count"])
        
        return {
            "input_size": input_size,
            "target_scale": target_scale,
            "target_size": (target_h, target_w),
            "recommended_strategy": strategies[0] if strategies else None,
            "all_strategies": strategies
        }


    @staticmethod
    def calculate_absolute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Calcula la diferencia absoluta entre dos imágenes"""
        try:
            # Asegurar mismo tamaño
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Convertir a float para evitar overflow
            img1_float = img1.astype(np.float32)
            img2_float = img2.astype(np.float32)
            
            # Calcular diferencia absoluta
            diff = np.abs(img1_float - img2_float)
            
            # Normalizar a rango [0, 255]
            diff_normalized = (diff / np.max(diff) * 255).astype(np.uint8)
            
            return diff_normalized
            
        except Exception as e:
            logger.error(f"Error calculando diferencia absoluta: {e}")
            return None

    @staticmethod
    def create_comparison_visualization(original: np.ndarray, enhanced: np.ndarray, 
                                    difference: np.ndarray = None) -> np.ndarray:
        """Crea visualización de comparación lado a lado"""
        try:
            # Asegurar mismo tamaño
            if original.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            
            # Calcular diferencia si no se proporciona
            if difference is None:
                difference = ImageUtils.calculate_absolute_difference(original, enhanced)
            
            # Crear visualización dependiendo del número de imágenes
            if difference is not None:
                # Tres imágenes: original, enhanced, difference
                comparison = np.hstack([original, enhanced, difference])
            else:
                # Dos imágenes: original, enhanced
                comparison = np.hstack([original, enhanced])
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error creando visualización de comparación: {e}")
            return None

    @staticmethod
    def generate_difference_heatmap(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Genera mapa de calor de las diferencias"""
        try:
            # Calcular diferencia absoluta
            diff = ImageUtils.calculate_absolute_difference(img1, img2)
            if diff is None:
                return None
            
            # Convertir a escala de grises si es necesario
            if len(diff.shape) == 3:
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            else:
                diff_gray = diff
            
            # Aplicar mapa de color (COLORMAP_JET para heatmap)
            heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Error generando mapa de calor: {e}")
            return None

    @staticmethod
    def analyze_difference_statistics(difference: np.ndarray) -> Dict[str, Any]:
        """Analiza estadísticas de la imagen de diferencia"""
        try:
            # Convertir a escala de grises si es necesario
            if len(difference.shape) == 3:
                diff_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            else:
                diff_gray = difference
            
            # Normalizar a [0, 1]
            diff_normalized = diff_gray.astype(np.float32) / 255.0
            
            stats = {
                "mean_difference": float(np.mean(diff_normalized)),
                "std_difference": float(np.std(diff_normalized)),
                "max_difference": float(np.max(diff_normalized)),
                "min_difference": float(np.min(diff_normalized)),
                "median_difference": float(np.median(diff_normalized)),
                "percentile_95": float(np.percentile(diff_normalized, 95)),
                "percentile_99": float(np.percentile(diff_normalized, 99))
            }
            
            # Análisis de distribución
            hist, bins = np.histogram(diff_normalized, bins=50, range=(0, 1))
            stats["histogram"] = {
                "values": hist.tolist(),
                "bins": bins.tolist()
            }
            
            # Clasificación de calidad basada en estadísticas
            if stats["mean_difference"] < 0.05:
                stats["quality_assessment"] = "Excelente - diferencias mínimas"
            elif stats["mean_difference"] < 0.1:
                stats["quality_assessment"] = "Muy buena - diferencias bajas"
            elif stats["mean_difference"] < 0.2:
                stats["quality_assessment"] = "Buena - diferencias moderadas"
            elif stats["mean_difference"] < 0.3:
                stats["quality_assessment"] = "Regular - diferencias notables"
            else:
                stats["quality_assessment"] = "Baja - diferencias significativas"
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analizando estadísticas de diferencia: {e}")
            return {}


# Instancia global de utilidades
image_utils = ImageUtils()