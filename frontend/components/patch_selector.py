"""
Componente para selección interactiva de parches
Maneja el canvas y la extracción de coordenadas
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
    """Selector interactivo de parches de imagen"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.max_canvas_width = 800
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
        st.markdown('<h3 class="sub-header">📊 Información de la Imagen</h3>', unsafe_allow_html=True)
        
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
    
    def show_patch_configuration(self, available_models: list, architecture: str) -> Optional[Dict[str, Any]]:
        """Muestra panel de configuración de parches"""
        st.markdown("**⚙️ Configuración del Parche:**")
        
        # Filtrar modelos por arquitectura seleccionada
        arch_models = [m for m in available_models 
                      if m["architecture"].upper() == architecture.upper() and m["available"]]
        
        if not arch_models:
            st.error(f"❌ No hay modelos disponibles para {architecture}")
            return None
        
        # Obtener recomendaciones
        recommendations = self.api_client.get_model_recommendations(architecture)
        
        # Configuración de tamaño de parche
        available_sizes = recommendations.get("available_sizes", [64, 128, 256, 512])
        if not available_sizes:
            available_sizes = [256]  # Fallback
        
        recommended_size = recommendations.get("recommended_start", 256)
        default_index = available_sizes.index(recommended_size) if recommended_size in available_sizes else 0
        
        patch_size = st.selectbox(
            "Tamaño del Parche:",
            available_sizes,
            index=default_index,
            help=f"Tamaños disponibles para {architecture}"
        )
        
        # Configuración de factor de escalado
        max_recommended_scale = min(recommendations.get("max_scale", 16), 16)
        scale_options = [2**i for i in range(1, int(np.log2(max_recommended_scale)) + 2) if 2**i <= 16]
        
        target_scale = st.selectbox(
            "Factor de Escalado:",
            scale_options,
            index=0,
            help=f"Factor de escalado (máximo recomendado: x{max_recommended_scale})"
        )
        
        # Validar factibilidad
        is_feasible, message = self.api_client.validate_upsampling_feasibility(
            architecture, patch_size, target_scale
        )
        
        if is_feasible:
            st.success(f"✅ {message}")
            
            # Mostrar ruta de procesamiento
            path_info = self.api_client.get_upsampling_path(architecture, patch_size, target_scale)
            if path_info:
                with st.expander("Ver ruta de procesamiento"):
                    st.markdown(f"**Pasos requeridos:** {len(path_info['path'])}")
                    for i, model_name in enumerate(path_info['path']):
                        st.markdown(f"**{i+1}.** {model_name}")
        else:
            st.error(f"❌ {message}")
            return None
        
        # Información adicional
        with st.expander("ℹ️ Información adicional"):
            st.markdown(f"""
            - **Arquitectura seleccionada:** {architecture}
            - **Modelos disponibles:** {len(arch_models)}
            - **Tamaño final esperado:** {patch_size * target_scale} × {patch_size * target_scale} px
            - **Tiempo estimado:** {self._estimate_processing_time(len(scale_options))} segundos
            """)
        
        return {
            "architecture": architecture,
            "patch_size": patch_size,
            "target_scale": target_scale,
            "path_info": path_info
        }
    
    def show_interactive_canvas(self, image: np.ndarray) -> Optional[Dict]:
        """Muestra canvas interactivo para selección de parches"""
        st.markdown('<h3 class="sub-header">🎯 Selección de Parche</h3>', unsafe_allow_html=True)
        
        # Calcular dimensiones del canvas
        canvas_width, canvas_height, scale_factor = self._calculate_canvas_dimensions(image)
        
        # Redimensionar imagen para mostrar
        display_image = self._prepare_display_image(image, canvas_width, canvas_height)
        
        # Instrucciones
        st.markdown("""
        <div class="instruction-text">
        🖱️ <strong>Instrucciones:</strong> Dibuja un rectángulo sobre la región que deseas procesar.
        El parche se ajustará automáticamente al tamaño configurado.
        </div>
        """, unsafe_allow_html=True)
        
        # Canvas interactivo
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",  # Naranja transparente
            stroke_width=3,
            stroke_color="#FF4500",  # Naranja sólido
            background_image=Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            width=canvas_width,
            height=canvas_height,
            drawing_mode="rect",
            key="patch_selector_canvas",
            display_toolbar=True
        )
        
        # Mostrar información de selección
        if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
            self._display_selection_info(canvas_result, image.shape, scale_factor)
        else:
            show_info_box("Dibuja un rectángulo para seleccionar el parche a procesar", "info")
        
        return canvas_result
    
    def extract_selection_coordinates(self, canvas_result: Dict, image: np.ndarray) -> Optional[Dict[str, int]]:
        """Extrae coordenadas de la selección del canvas"""
        if not canvas_result.json_data or not canvas_result.json_data["objects"]:
            st.warning("⚠️ No hay selección. Dibuja un rectángulo primero.")
            return None
        
        try:
            # Obtener el último rectángulo dibujado
            rect = canvas_result.json_data["objects"][-1]
            
            # Calcular factor de escala
            canvas_width, canvas_height, scale_factor = self._calculate_canvas_dimensions(image)
            
            # Coordenadas en la imagen original
            x = int(rect["left"] / scale_factor)
            y = int(rect["top"] / scale_factor)
            width = int(rect["width"] / scale_factor)
            height = int(rect["height"] / scale_factor)
            
            # Validar coordenadas
            img_height, img_width = image.shape[:2]
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            width = min(width, img_width - x)
            height = min(height, img_height - y)
            
            return {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }
            
        except Exception as e:
            st.error(f"Error extrayendo coordenadas: {e}")
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
    
    def _display_selection_info(self, canvas_result: Dict, image_shape: Tuple, scale_factor: float):
        """Muestra información de la selección actual"""
        try:
            rect = canvas_result.json_data["objects"][-1]
            
            # Coordenadas en imagen original
            x = int(rect["left"] / scale_factor)
            y = int(rect["top"] / scale_factor)
            width = int(rect["width"] / scale_factor)
            height = int(rect["height"] / scale_factor)
            
            # Validar coordenadas
            img_height, img_width = image_shape[:2]
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            width = min(width, img_width - x)
            height = min(height, img_height - y)
            
            # Mostrar información
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **📍 Coordenadas:**
                - X: {x} px
                - Y: {y} px
                """)
            with col2:
                st.markdown(f"""
                **📏 Dimensiones:**
                - Ancho: {width} px
                - Alto: {height} px
                """)
            
            # Advertencias si es necesario
            if width != height:
                st.warning(f"⚠️ El parche se cuadrará automáticamente (se usará el menor: {min(width, height)}px)")
            
            if width < 64 or height < 64:
                st.error("❌ El parche es muy pequeño (mínimo 64x64 px)")
                
        except Exception as e:
            logger.error(f"Error mostrando info de selección: {e}")
    
    def _estimate_processing_time(self, num_steps: int) -> float:
        """Estima tiempo de procesamiento basado en número de pasos"""
        # Tiempo base por paso (estimado)
        base_time_per_step = 2.0  # segundos
        return num_steps * base_time_per_step
    
    def preview_patch(self, image: np.ndarray, coordinates: Dict[str, int], target_size: int) -> Optional[np.ndarray]:
        """Genera vista previa del parche que será procesado"""
        try:
            x, y = coordinates["x"], coordinates["y"]
            width, height = coordinates["width"], coordinates["height"]
            
            # Extraer parche
            patch = image[y:y+height, x:x+width]
            
            # Redimensionar al tamaño objetivo si es necesario
            if patch.shape[0] != target_size or patch.shape[1] != target_size:
                patch = cv2.resize(patch, (target_size, target_size))
            
            return patch
            
        except Exception as e:
            logger.error(f"Error generando preview: {e}")
            return None