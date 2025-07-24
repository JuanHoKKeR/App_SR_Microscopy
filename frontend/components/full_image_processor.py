"""
Componente para procesamiento de imagen completa
"""

import streamlit as st
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Tuple
import logging

from .api_client import APIClient
from .ui_config import show_info_box, show_metric_card, show_progress_steps

logger = logging.getLogger(__name__)

class FullImageProcessor:
    """Procesador de imagen completa con estrategias automáticas"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def show_image_upload_section(self) -> Optional[Tuple[np.ndarray, Any]]:
        """Sección de carga de imagen"""
        st.markdown('<h3 class="sub-header">📁 Cargar Imagen</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Selecciona imagen para procesamiento completo:",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="La imagen será procesada completamente según la estrategia seleccionada"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # Mostrar información básica
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    show_metric_card(f"{image_np.shape[1]}", "Ancho (px)")
                with col2:
                    show_metric_card(f"{image_np.shape[0]}", "Alto (px)")
                with col3:
                    channels = image_np.shape[2] if len(image_np.shape) == 3 else 1
                    show_metric_card(f"{channels}", "Canales")
                with col4:
                    size_mb = (uploaded_file.size / (1024 * 1024))
                    show_metric_card(f"{size_mb:.1f} MB", "Tamaño")
                
                # Mostrar imagen
                st.image(image, caption=f"Imagen original: {uploaded_file.name}", use_column_width=True)
                
                return image_np, uploaded_file
                
            except Exception as e:
                st.error(f"Error cargando imagen: {e}")
                return None, None
        
        return None, None
    
    def show_processing_configuration(self, image_shape: Tuple[int, int], 
                                    available_models: list, architecture: str) -> Optional[Dict[str, Any]]:
        """Configuración de procesamiento"""
        st.markdown('<h3 class="sub-header">⚙️ Configuración de Procesamiento</h3>', unsafe_allow_html=True)
        
        # Configuración básica
        col1, col2 = st.columns(2)
        
        with col1:
            target_scale = st.selectbox(
                "Factor de Escalado:",
                [2, 4, 8, 16],
                help="Factor de escalado total deseado"
            )
            
            strategy_type = st.radio(
                "Estrategia de Procesamiento:",
                ["🤖 Automática", "🖼️ Imagen Completa", "🧩 División en Parches"],
                help="Automática: el sistema elige la mejor estrategia"
            )
        
        with col2:
            # Obtener estrategias disponibles
            strategies = self.api_client.get_processing_strategies(
                image_shape[1], image_shape[0], target_scale, architecture
            )
            
            if strategies:
                self._show_strategies_info(strategies)
            else:
                st.warning("⚠️ No se pudieron obtener estrategias de procesamiento")
                return None
        
        # Configuración avanzada para parches
        advanced_config = {}
        if "Parches" in strategy_type:
            with st.expander("🔧 Configuración Avanzada de Parches"):
                patch_size = st.selectbox(
                    "Tamaño de Parche:",
                    [128, 256, 512],
                    index=1,
                    help="Tamaño de cada parche individual"
                )
                
                overlap = st.slider(
                    "Overlap entre Parches:",
                    min_value=0,
                    max_value=64,
                    value=32,
                    help="Overlap para suavizar transiciones"
                )
                
                advanced_config = {"patch_size": patch_size, "overlap": overlap}
        
        # Opciones adicionales
        with st.expander("📊 Opciones Adicionales"):
            evaluate_quality = st.checkbox(
                "🧠 Evaluar calidad con KimiaNet",
                value=False,
                help="Calcular métricas de calidad (toma más tiempo)"
            )
        
        # Mapear estrategia seleccionada
        strategy_map = {
            "🤖 Automática": "automatic",
            "🖼️ Imagen Completa": "full_image", 
            "🧩 División en Parches": "patch_based"
        }
        
        return {
            "target_scale": target_scale,
            "strategy_type": strategy_map[strategy_type],
            "architecture": architecture,
            "evaluate_quality": evaluate_quality,
            **advanced_config
        }
    
    def _show_strategies_info(self, strategies: Dict[str, Any]):
        """Muestra información sobre estrategias disponibles"""
        st.markdown("**📋 Estrategias Disponibles:**")
        
        recommended = strategies.get("recommended_strategy")
        if recommended:
            st.success(f"✅ **Recomendada:** {recommended['description']}")
            
            if recommended["type"] == "patch_based":
                st.info(f"📊 Se usarán {recommended['patch_count']} parches de {recommended['patch_size']}x{recommended['patch_size']}")
        
        all_strategies = strategies.get("all_strategies", [])
        if len(all_strategies) > 1:
            with st.expander("Ver todas las estrategias"):
                for i, strategy in enumerate(all_strategies):
                    status = "🟢" if strategy.get("memory_efficient", True) else "🟡"
                    st.markdown(f"{status} **Opción {i+1}:** {strategy['description']}")
    
    def process_image(self, uploaded_file, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Procesa la imagen con la configuración dada"""
        
        # Mostrar progreso
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            st.markdown('<h3 class="sub-header">🔄 Procesando Imagen Completa</h3>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Paso 1: Iniciando
            status_text.text("🚀 Enviando imagen para procesamiento...")
            progress_bar.progress(10)
            
            # Resetear archivo
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            
            # Llamar a la API
            result = self.api_client.process_full_image(
                image_file=uploaded_file,
                **config
            )
            
            if not result:
                st.error("❌ Error en el procesamiento")
                return None
            
            # Simular progreso
            for i, step in enumerate(["Analizando imagen", "Aplicando estrategia", "Procesando", "Reconstruyendo"], 2):
                progress_bar.progress(20 + (i * 20))
                status_text.text(f"⚙️ {step}...")
                
            progress_bar.progress(100)
            status_text.text("✅ ¡Procesamiento completado!")
            
            # Limpiar después de un momento
            import time
            time.sleep(1)
            progress_container.empty()
            status_container.empty()
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error durante el procesamiento: {e}")
            logger.error(f"Error en procesamiento: {e}")
            return None
    
    def show_processing_preview(self, config: Dict[str, Any], image_shape: Tuple[int, int]):
        """Muestra vista previa del procesamiento"""
        st.markdown('<h3 class="sub-header">📋 Resumen del Procesamiento</h3>', unsafe_allow_html=True)
        
        h, w = image_shape[:2]
        target_scale = config["target_scale"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Configuración:**")
            st.markdown(f"- **Imagen:** {w} × {h} píxeles")
            st.markdown(f"- **Arquitectura:** {config['architecture']}")
            st.markdown(f"- **Factor de escala:** ×{target_scale}")
            st.markdown(f"- **Estrategia:** {config['strategy_type'].title()}")
        
        with col2:
            st.markdown("**📊 Resultado Esperado:**")
            target_w, target_h = w * target_scale, h * target_scale
            st.markdown(f"- **Resolución final:** {target_w} × {target_h} píxeles")
            st.markdown(f"- **Aumento de resolución:** {target_scale**2}x píxeles")
            
            # Estimación de tiempo
            if config["strategy_type"] == "patch_based":
                estimated_time = config.get("patch_size", 256) // 64 * 3  # Estimación rough
                st.markdown(f"- **Tiempo estimado:** ~{estimated_time}s")
            else:
                st.markdown("- **Tiempo estimado:** ~5-15s")
        
        # Advertencias
        if target_w > 4096 or target_h > 4096:
            st.warning("⚠️ La imagen resultante será muy grande (>4K). El procesamiento puede tomar tiempo.")
        
        if config["strategy_type"] == "patch_based":
            st.info("💡 Se usará división en parches para manejar la imagen de forma eficiente.")