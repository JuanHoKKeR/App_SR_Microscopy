"""
Componente corregido para selección interactiva de parches
Corrige problemas de selección de coordenadas y configuración por escala
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from typing import Optional, Dict, Any, Tuple
import logging

from .ui_config import show_info_box, show_metric_card
from .api_client import APIClient

logger = logging.getLogger(__name__)

class PatchSelector:
    """Selector interactivo de parches corregido"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.max_canvas_width = 600
        self.max_canvas_height = 600
    
    def load_and_display_image(self, uploaded_file) -> Optional[np.ndarray]:
        """Carga y muestra información de la imagen subida"""
        try:
            # Cargar imagen
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convertir a BGR si es necesario
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # PIL usa RGB, convertir a BGR para OpenCV
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Mostrar información de la imagen
            self._display_image_info(image_np, uploaded_file.name)
            
            return image_np
            
        except Exception as e:
            st.error(f"Error cargando imagen: {e}")
            return None
    
    def _display_image_info(self, image: np.ndarray, filename: str):
        """Muestra información detallada de la imagen"""
        st.markdown('<h4>📊 Información de la Imagen</h4>', unsafe_allow_html=True)
        
        # Métricas en columnas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_metric_card(f"{image.shape[1]}", "Ancho (px)")
        with col2:
            show_metric_card(f"{image.shape[0]}", "Alto (px)")
        with col3:
            channels = image.shape[2] if len(image.shape) == 3 else 1
            show_metric_card(f"{channels}", "Canales")
        with col4:
            size_mb = (image.nbytes / (1024 * 1024))
            show_metric_card(f"{size_mb:.1f} MB", "Tamaño")
        
        # Información adicional
        show_info_box(f"""
        **Archivo:** {filename}<br>
        **Resolución:** {image.shape[1]} × {image.shape[0]} píxeles<br>
        **Tipo:** {'Color' if len(image.shape) == 3 else 'Escala de grises'}<br>
        **Formato interno:** {image.dtype}
        """)
    
    def show_scale_configuration(self, available_models: list, architecture: str) -> Optional[Dict[str, Any]]:
        """Muestra configuración basada en factor de escala en lugar de modelo específico"""
        st.markdown("**⚙️ Configuración de Procesamiento:**")
        
        # Filtrar modelos por arquitectura seleccionada
        arch_models = [m for m in available_models 
                      if m["architecture"].upper() == architecture.upper() and m["available"]]
        
        if not arch_models:
            st.error(f"❌ No hay modelos disponibles para {architecture}")
            return None
        
        # Obtener tamaños de parche disponibles
        available_patch_sizes = sorted(list(set([m["input_size"] for m in arch_models])))
        
        # Configuración de tamaño de parche
        col1, col2 = st.columns(2)
        
        with col1:
            patch_size = st.selectbox(
                "📐 Tamaño de Parche:",
                available_patch_sizes,
                index=len(available_patch_sizes)-1 if available_patch_sizes else 0,
                help=f"Tamaños disponibles para {architecture}",
                key="patch_size_select"
            )
        
        with col2:
            # Calcular escalas máximas posibles desde este tamaño de parche
            max_possible_scales = self._calculate_max_scale(patch_size, arch_models)
            
            # Opciones de escala disponibles
            scale_options = []
            for scale in [2, 4, 8, 16]:
                if scale <= max_possible_scales:
                    scale_options.append(scale)
            
            if not scale_options:
                st.error("❌ No hay escalas disponibles para este tamaño de parche")
                return None
            
            target_scale = st.selectbox(
                "🔍 Factor de Escala:",
                scale_options,
                help=f"Escala máxima disponible: x{max_possible_scales}",
                key="target_scale_select"
            )
        
        # Validar y obtener ruta de procesamiento
        processing_path = self._get_processing_path(patch_size, target_scale, architecture, arch_models)
        
        if not processing_path:
            st.error(f"❌ No se puede alcanzar x{target_scale} desde {patch_size}px con {architecture}")
            return None
        
        # Mostrar información del procesamiento
        self._show_processing_info(processing_path, patch_size, target_scale, architecture)
        
        return {
            "architecture": architecture,
            "patch_size": patch_size,
            "target_scale": target_scale,
            "processing_path": processing_path,
            "final_size": patch_size * target_scale
        }
    
    def _calculate_max_scale(self, patch_size: int, arch_models: list) -> int:
        """Calcula la escala máxima posible desde un tamaño de parche dado"""
        max_scale = 1
        current_size = patch_size
        
        while True:
            next_size = current_size * 2
            # Buscar si existe un modelo que vaya de current_size a next_size
            model_exists = any(
                m["input_size"] == current_size and m["output_size"] == next_size 
                for m in arch_models
            )
            
            if model_exists:
                max_scale *= 2
                current_size = next_size
            else:
                break
        
        return max_scale
    
    def _get_processing_path(self, start_size: int, target_scale: int, architecture: str, arch_models: list) -> list:
        """Obtiene la ruta de modelos necesarios para alcanzar la escala objetivo"""
        path = []
        current_size = start_size
        target_size = start_size * target_scale
        
        while current_size < target_size:
            next_size = current_size * 2
            
            # Buscar modelo que vaya de current_size a next_size
            model_found = None
            for model in arch_models:
                if model["input_size"] == current_size and model["output_size"] == next_size:
                    model_found = model["name"]
                    break
            
            if model_found:
                path.append(model_found)
                current_size = next_size
            else:
                break
        
        return path if current_size == target_size else []
    
    def _show_processing_info(self, processing_path: list, patch_size: int, target_scale: int, architecture: str):
        """Muestra información detallada del procesamiento que se realizará"""
        with st.expander("ℹ️ Detalles del Procesamiento"):
            st.markdown(f"**🏗️ Arquitectura:** {architecture}")
            st.markdown(f"**📐 Tamaño inicial:** {patch_size} × {patch_size} px")
            st.markdown(f"**🔍 Factor de escala:** ×{target_scale}")
            st.markdown(f"**📏 Tamaño final:** {patch_size * target_scale} × {patch_size * target_scale} px")
            st.markdown(f"**🔄 Pasos requeridos:** {len(processing_path)}")
            
            st.markdown("**📋 Secuencia de modelos:**")
            current_size = patch_size
            for i, model_name in enumerate(processing_path):
                next_size = current_size * 2
                st.markdown(f"   **{i+1}.** {model_name}: {current_size}×{current_size} → {next_size}×{next_size}")
                current_size = next_size
    
    def show_interactive_canvas(self, image: np.ndarray, patch_size: int) -> Optional[Dict]:
        """Muestra canvas interactivo para selección de parches"""
        
        # Calcular dimensiones del canvas
        canvas_width, canvas_height, scale_factor = self._calculate_canvas_dimensions(image)
        
        # Redimensionar imagen para mostrar
        display_image = self._prepare_display_image(image, canvas_width, canvas_height)
        
        # Instrucciones
        st.markdown(f"""
        <div class="instruction-text">
        🖱️ <strong>Instrucciones:</strong> Dibuja un rectángulo sobre la región que deseas procesar.
        El parche se ajustará automáticamente a {patch_size}×{patch_size} píxeles.
        </div>
        """, unsafe_allow_html=True)
        
        # Canvas interactivo
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Naranja transparente
            stroke_width=2,
            stroke_color="#FF4500",  # Naranja sólido
            background_image=Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            width=canvas_width,
            height=canvas_height,
            drawing_mode="rect",
            key="patch_canvas",
            display_toolbar=True
        )
        
        # Mostrar información de selección
        if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
            self._display_selection_info(canvas_result, image.shape, scale_factor, patch_size)
        else:
            show_info_box(f"Dibuja un rectángulo para seleccionar el parche de {patch_size}×{patch_size} píxeles", "info")
        
        return canvas_result
    
    def extract_selection_coordinates(self, canvas_result: Dict, image: np.ndarray, patch_size: int) -> Optional[Dict[str, int]]:
        """Extrae coordenadas de la selección del canvas y las ajusta al tamaño de parche"""
        if not canvas_result.json_data or not canvas_result.json_data["objects"]:
            st.warning("⚠️ No hay selección. Dibuja un rectángulo primero.")
            return None
        
        try:
            # Obtener el último rectángulo dibujado
            rect = canvas_result.json_data["objects"][-1]
            
            # Calcular factor de escala
            canvas_width, canvas_height, scale_factor = self._calculate_canvas_dimensions(image)
            
            # Coordenadas del centro del rectángulo en la imagen original
            center_x = int((rect["left"] + rect["width"]/2) / scale_factor)
            center_y = int((rect["top"] + rect["height"]/2) / scale_factor)
            
            # Calcular coordenadas del parche centrado
            half_patch = patch_size // 2
            x = max(0, min(center_x - half_patch, image.shape[1] - patch_size))
            y = max(0, min(center_y - half_patch, image.shape[0] - patch_size))
            
            # Asegurar que el parche esté completamente dentro de la imagen
            if x + patch_size > image.shape[1]:
                x = image.shape[1] - patch_size
            if y + patch_size > image.shape[0]:
                y = image.shape[0] - patch_size
            
            return {
                "x": x,
                "y": y,
                "width": patch_size,
                "height": patch_size
            }
            
        except Exception as e:
            st.error(f"Error extrayendo coordenadas: {e}")
            logger.error(f"Error extrayendo coordenadas: {e}")
            return None
    
    def _calculate_canvas_dimensions(self, image: np.ndarray) -> Tuple[int, int, float]:
        """Calcula dimensiones óptimas del canvas"""
        img_height, img_width = image.shape[:2]
        
        # Calcular factor de escala para ajustar al canvas
        width_scale = self.max_canvas_width / img_width
        height_scale = self.max_canvas_height / img_height
        scale_factor = min(width_scale, height_scale, 1.0)  # No ampliar
        
        canvas_width = int(img_width * scale_factor)
        canvas_height = int(img_height * scale_factor)
        
        return canvas_width, canvas_height, scale_factor
    
    def _prepare_display_image(self, image: np.ndarray, canvas_width: int, canvas_height: int) -> np.ndarray:
        """Prepara imagen para mostrar en el canvas"""
        if image.shape[1] != canvas_width or image.shape[0] != canvas_height:
            display_image = cv2.resize(image, (canvas_width, canvas_height))
        else:
            display_image = image.copy()
        
        return display_image
    
    def _display_selection_info(self, canvas_result: Dict, image_shape: Tuple, scale_factor: float, patch_size: int):
        """Muestra información de la selección actual"""
        try:
            rect = canvas_result.json_data["objects"][-1]
            
            # Coordenadas del centro del rectángulo en imagen original
            center_x = int((rect["left"] + rect["width"]/2) / scale_factor)
            center_y = int((rect["top"] + rect["height"]/2) / scale_factor)
            
            # Calcular coordenadas del parche final
            half_patch = patch_size // 2
            x = max(0, min(center_x - half_patch, image_shape[1] - patch_size))
            y = max(0, min(center_y - half_patch, image_shape[0] - patch_size))
            
            # Mostrar información
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **📍 Coordenadas del Parche:**
                - X: {x} px
                - Y: {y} px
                """)
            with col2:
                st.markdown(f"""
                **📏 Dimensiones:**
                - Ancho: {patch_size} px
                - Alto: {patch_size} px
                """)
            
            # Mostrar área seleccionada como porcentaje
            total_area = image_shape[0] * image_shape[1]
            patch_area = patch_size * patch_size
            percentage = (patch_area / total_area) * 100
            
            st.info(f"📊 El parche representa el {percentage:.2f}% del área total de la imagen")
                
        except Exception as e:
            logger.error(f"Error mostrando info de selección: {e}")
    
    def preview_patch(self, image: np.ndarray, coordinates: Dict[str, int]) -> Optional[np.ndarray]:
        """Genera vista previa del parche que será procesado"""
        try:
            x, y = coordinates["x"], coordinates["y"]
            width, height = coordinates["width"], coordinates["height"]
            
            # Extraer parche
            patch = image[y:y+height, x:x+width]
            
            return patch
            
        except Exception as e:
            logger.error(f"Error generando preview: {e}")
            return None