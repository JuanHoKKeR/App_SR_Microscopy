"""
Utilidades para manejo de imágenes
Conversiones, validaciones y operaciones básicas
"""

import numpy as np
import cv2
import base64
import io
from PIL import Image
from typing import Tuple, Optional, Union
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

# Instancia global de utilidades
image_utils = ImageUtils()