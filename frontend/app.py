#!/usr/bin/env python3
"""
Frontend Streamlit para aplicación de superresolución
Aplicación principal con navegación por funcionalidades
"""

import streamlit as st
import sys
from pathlib import Path

# Agregar directorio de componentes al path
sys.path.append(str(Path(__file__).parent))

# Importar configuración base
from components.ui_config import setup_page_config, load_custom_css
from components.api_client import APIClient

# Importar páginas/funcionalidades
from pages.funcionalidad_1 import funcionalidad_1_page
from pages.funcionalidad_2 import funcionalidad_2_page  
from pages.funcionalidad_3 import funcionalidad_3_page
from utils.session_state import init_session_state

def main():
    """Función principal de la aplicación"""
    
    # Configurar página
    setup_page_config()
    load_custom_css()
    
    # Inicializar estado de sesión
    init_session_state()
    
    # Header principal
    st.markdown('<h1 class="main-header">🔬 Microscopy Super-Resolution</h1>', unsafe_allow_html=True)
    
    # Verificar conexión con API
    api_client = APIClient()
    if not api_client.check_connection():
        st.error("❌ No se puede conectar con la API. Asegúrate de que el backend esté ejecutándose.")
        st.info("🔧 **Instrucciones para iniciar el backend:**")
        st.code("cd backend && python main.py", language="bash")
        
        # Mostrar estado de conexión en sidebar
        with st.sidebar:
            st.markdown("### 🔌 Estado de Conexión")
            st.error("Backend desconectado")
        
        st.stop()
    
    # Sidebar con navegación principal
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📋 Funcionalidades</h2>', unsafe_allow_html=True)
        
        # Mostrar estado de conexión
        st.success("✅ Backend conectado")
        
        # Navegación principal
        funcionalidades = {
            "🎯 Funcionalidad 1": "Selección de Parches y Super-Resolución",
            "🖼️ Funcionalidad 2": "Procesamiento de Imagen Completa", 
            "📊 Funcionalidad 3": "Evaluación y Comparación de Modelos"
        }
        
        selected_func = st.radio(
            "Selecciona una funcionalidad:",
            list(funcionalidades.keys()),
            help="Cada funcionalidad tiene propósitos específicos"
        )
        
        # Mostrar descripción de la funcionalidad seleccionada
        st.markdown("---")
        st.markdown("**Descripción:**")
        st.info(funcionalidades[selected_func])
        
        # Información de modelos disponibles
        st.markdown("---")
        show_model_status_sidebar(api_client)
    
    # Renderizar funcionalidad seleccionada
    if selected_func == "🎯 Funcionalidad 1":
        funcionalidad_1_page(api_client)
    elif selected_func == "🖼️ Funcionalidad 2":
        funcionalidad_2_page(api_client)
    elif selected_func == "📊 Funcionalidad 3":
        funcionalidad_3_page(api_client)

def show_model_status_sidebar(api_client):
    """Muestra estado resumido de modelos en sidebar"""
    available_models = api_client.get_available_models()
    
    if not available_models:
        st.error("❌ No se pudieron cargar modelos")
        return
    
    st.markdown("**🤖 Estado de Modelos:**")
    
    # Contar por arquitectura
    architectures = {}
    for model in available_models:
        arch = model["architecture"]
        if arch not in architectures:
            architectures[arch] = {"available": 0, "total": 0}
        
        architectures[arch]["total"] += 1
        if model["available"]:
            architectures[arch]["available"] += 1
    
    # Mostrar conteo por arquitectura
    for arch, stats in architectures.items():
        available = stats["available"]
        total = stats["total"]
        
        if available == total:
            icon = "✅"
        elif available > 0:
            icon = "⚠️"
        else:
            icon = "❌"
        
        percentage = (available/total)*100 if total > 0 else 0
        st.markdown(f"{icon} **{arch}**: {available}/{total} ({percentage:.0f}%)")
    
    # Botón para ver detalles
    if st.button("🔍 Ver detalles de modelos"):
        st.session_state.show_model_details = True
    
    # Mostrar detalles si se solicita
    if st.session_state.get("show_model_details", False):
        with st.expander("📋 Detalles de modelos", expanded=True):
            for model in available_models:
                status = "✅" if model["available"] else "❌"
                st.markdown(f"{status} **{model['name']}**")
                st.markdown(f"   └─ {model['input_size']}→{model['output_size']}px ({model['architecture']})")
            
            if st.button("❌ Cerrar detalles"):
                st.session_state.show_model_details = False
                
                
def handle_full_image_mode(api_client, available_models, architecture):
    """Maneja el modo de imagen completa"""
    st.markdown('<h2 class="sub-header">🖼️ Procesamiento de Imagen Completa</h2>', unsafe_allow_html=True)
    
    # Importar componente
    from components.full_image_processor import FullImageProcessor
    from components.results_viewer import ResultsViewer
    
    # Inicializar procesador
    full_processor = FullImageProcessor(api_client)
    results_viewer = ResultsViewer()
    
    # Sección de carga de imagen
    image_data, uploaded_file = full_processor.show_image_upload_section()
    
    if image_data is not None and uploaded_file is not None:
        # Configuración de procesamiento
        config = full_processor.show_processing_configuration(
            image_data.shape, available_models, architecture
        )
        
        if config is not None:
            # Vista previa del procesamiento
            full_processor.show_processing_preview(config, image_data.shape)
            
            # Botón de procesamiento
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                process_button = st.button(
                    "🚀 Procesar Imagen Completa",
                    type="primary",
                    use_container_width=True
                )
            
            # Procesar si se hace clic
            if process_button:
                result = full_processor.process_image(uploaded_file, config)
                
                if result and result.get("success"):
                    # Mostrar resultados
                    st.markdown("---")
                    results_viewer.display_full_image_results(result)
                else:
                    st.error("❌ Error en el procesamiento de la imagen")
    
    else:
        # Información sobre la funcionalidad
        show_info_box("""
        **🖼️ Procesamiento de Imagen Completa**<br><br>
        Esta funcionalidad permite procesar imágenes completas de cualquier tamaño usando:<br>
        • **Estrategia automática**: El sistema elige la mejor forma de procesar<br>
        • **Imagen completa**: Para imágenes pequeñas (recomendado <1024px)<br>
        • **División en parches**: Para imágenes grandes con reconstrucción inteligente<br><br>
        **Ventajas:**<br>
        • Manejo automático de imágenes grandes<br>
        • Reconstrucción suave sin artefactos<br>
        • Múltiples estrategias según el tamaño<br>
        • Escalado hasta 16x con múltiples modelos
        """, "info")

if __name__ == "__main__":
    main()