#!/usr/bin/env python3
"""
Frontend Streamlit para aplicación de superresolución
Versión modular con componentes separados
"""

import streamlit as st
import sys
from pathlib import Path

# Agregar directorio de componentes al path
sys.path.append(str(Path(__file__).parent))

from components.ui_config import setup_page_config, load_custom_css
from components.api_client import APIClient
from components.patch_selector import PatchSelector
from components.image_processor import ImageProcessorUI
from components.results_viewer import ResultsViewer
from utils.session_state import init_session_state, get_session_state, update_session_state

def main():
    """Función principal de la aplicación"""
    
    # Configurar página
    setup_page_config()
    load_custom_css()
    
    # Inicializar estado de sesión
    init_session_state()
    
    # Header principal
    st.markdown('<h1 class="main-header">🔬 Microscopy Super-Resolution</h1>', unsafe_allow_html=True)
    
    # Inicializar cliente API
    api_client = APIClient()
    
    # Verificar conexión con API
    if not api_client.check_connection():
        st.error("❌ No se puede conectar con la API. Asegúrate de que el backend esté ejecutándose.")
        st.info("🔧 **Instrucciones:**")
        st.code("cd backend && python main.py", language="bash")
        st.stop()
    
    # Obtener modelos disponibles
    available_models = api_client.get_available_models()
    if not available_models:
        st.error("❌ No se pudieron cargar los modelos disponibles")
        st.stop()
    
    # Sidebar con configuración
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Configuración</h2>', unsafe_allow_html=True)
        
        # Mostrar estado de modelos
        show_model_status(available_models)
        
        st.markdown("---")
        
        # Selección de modo
        mode = st.radio(
            "Modo de Procesamiento:",
            ["🎯 Selección de Parches", "🖼️ Imagen Completa"],
            help="Selecciona el modo de procesamiento que deseas usar"
        )
        
        update_session_state("mode", mode)
        
        st.markdown("---")
        
        # Configuración de arquitectura
        architectures = list(set([model["architecture"] for model in available_models if model["available"]]))
        
        if not architectures:
            st.error("No hay modelos disponibles")
            st.stop()
        
        architecture = st.selectbox(
            "Arquitectura:",
            architectures,
            help="Selecciona la arquitectura de red neuronal"
        )
        
        update_session_state("architecture", architecture)
    
    # Área principal
    if mode == "🎯 Selección de Parches":
        handle_patch_selection_mode(api_client, available_models, architecture)
    else:
        handle_full_image_mode(api_client, available_models, architecture)

def show_model_status(available_models):
    """Muestra el estado de los modelos disponibles"""
    st.markdown("**Estado de Modelos:**")
    
    # Agrupar por arquitectura
    architectures = {}
    for model in available_models:
        arch = model["architecture"]
        if arch not in architectures:
            architectures[arch] = {"available": 0, "total": 0}
        
        architectures[arch]["total"] += 1
        if model["available"]:
            architectures[arch]["available"] += 1
    
    # Mostrar estado por arquitectura
    for arch, stats in architectures.items():
        available = stats["available"]
        total = stats["total"]
        
        if available == total:
            icon = "✅"
            color = "green"
        elif available > 0:
            icon = "⚠️"
            color = "orange"
        else:
            icon = "❌"
            color = "red"
        
        st.markdown(f"{icon} **{arch}**: {available}/{total} modelos")
    
    # Mostrar detalles expandibles
    with st.expander("Ver detalles de modelos"):
        for model in available_models:
            status = "✅" if model["available"] else "❌"
            st.markdown(f"{status} {model['name']} ({model['input_size']}→{model['output_size']})")

def handle_patch_selection_mode(api_client, available_models, architecture):
    """Maneja el modo de selección de parches"""
    st.markdown('<h2 class="sub-header">📁 Cargar Imagen</h2>', unsafe_allow_html=True)
    
    # Subida de archivo
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de microscopía:",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Formatos soportados: PNG, JPG, JPEG, TIFF, BMP"
    )
    
    if uploaded_file is not None:
        # Procesar imagen subida
        patch_selector = PatchSelector(api_client)
        image_processor = ImageProcessorUI(api_client)
        results_viewer = ResultsViewer()
        
        # Cargar y mostrar información de imagen
        image = patch_selector.load_and_display_image(uploaded_file)
        
        if image is not None:
            # Configuración de procesamiento
            col1, col2 = st.columns([2, 1])
            
            with col2:
                # Panel de configuración
                config = patch_selector.show_patch_configuration(available_models, architecture)
                
                if config is None:
                    return
                
                # Botón de procesamiento
                process_button = st.button("🚀 Procesar Parche", type="primary")
            
            with col1:
                # Canvas interactivo
                canvas_result = patch_selector.show_interactive_canvas(image)
                
                # Procesar si hay selección y se hace clic
                if canvas_result and process_button:
                    selection = patch_selector.extract_selection_coordinates(canvas_result, image)
                    
                    if selection:
                        # Procesar con upsampling secuencial
                        with st.spinner("Procesando parche..."):
                            result = image_processor.process_sequential_upsampling(
                                uploaded_file, 
                                selection, 
                                config
                            )
                            
                            if result:
                                # Mostrar resultados
                                results_viewer.display_sequential_results(result)

def handle_full_image_mode(api_client, available_models, architecture):
    """Maneja el modo de imagen completa"""
    st.markdown('<h2 class="sub-header">🖼️ Procesamiento de Imagen Completa</h2>', unsafe_allow_html=True)
    
    st.info("🚧 **Funcionalidad en desarrollo**")
    st.markdown("""
    Esta funcionalidad incluirá:
    - Procesamiento automático por parches
    - Estrategias de división inteligente  
    - Manejo de overlap para evitar artefactos
    - Procesamiento en lotes
    """)
    
    # Panel de configuración placeholder
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Configuración:**")
        target_resolution = st.selectbox(
            "Resolución objetivo:",
            ["2048x2048", "4096x4096", "8192x8192"],
            help="Selecciona la resolución final deseada"
        )
        
        processing_strategy = st.radio(
            "Estrategia de procesamiento:",
            ["Automática (recomendada)", "Manual"],
            help="Automática: el sistema elige la mejor ruta"
        )
    
    with col2:
        st.markdown("**Estado:**")
        st.markdown("- 🚧 División automática por parches")
        st.markdown("- 🚧 Manejo de overlapping")
        st.markdown("- 🚧 Reconstrucción inteligente")
        st.markdown(f"- ✅ Arquitectura: {architecture}")
        
        if st.button("🚀 Procesar Imagen Completa", type="primary"):
            st.warning("⏳ Funcionalidad próximamente disponible")

if __name__ == "__main__":
    main()