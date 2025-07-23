#!/usr/bin/env python3
"""
Frontend Streamlit para aplicación de superresolución
Sistema de navegación independiente con funcionalidades separadas
"""

import streamlit as st
import sys
from pathlib import Path

# Agregar directorio de componentes al path
sys.path.append(str(Path(__file__).parent))

from components.ui_config import setup_page_config, load_custom_css
from components.api_client import APIClient
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
    
    # Sistema de navegación principal
    st.markdown("---")
    
    # Crear tabs para las funcionalidades
    tab1, tab2, tab3 = st.tabs([
        "🎯 Funcionalidad 1: Selección de Parches", 
        "🖼️ Funcionalidad 2: Imagen Completa", 
        "📊 Funcionalidad 3: Evaluación Comparativa"
    ])
    
    with tab1:
        handle_functionality_1(api_client, available_models)
    
    with tab2:
        handle_functionality_2(api_client, available_models)
    
    with tab3:
        handle_functionality_3(api_client, available_models)

def show_model_status_sidebar(available_models):
    """Muestra el estado de los modelos en el sidebar"""
    with st.sidebar:
        st.markdown('<h3 class="sub-header">📊 Estado de Modelos</h3>', unsafe_allow_html=True)
        
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

def handle_functionality_1(api_client, available_models):
    """Maneja la Funcionalidad 1: Selección de Parches"""
    st.markdown('<h2 class="sub-header">🎯 Selección de Parches y Super-Resolución</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Instrucciones:**
    1. Sube una imagen de microscopía
    2. Selecciona un parche dibujando un rectángulo
    3. Elige la arquitectura y factor de escala deseado
    4. Visualiza el resultado con métricas de calidad
    """)
    
    # Mostrar estado de modelos en sidebar
    show_model_status_sidebar(available_models)
    
    # Subida de archivo
    uploaded_file = st.file_uploader(
        "📁 Selecciona una imagen de microscopía:",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Formatos soportados: PNG, JPG, JPEG, TIFF, BMP",
        key="func1_upload"
    )
    
    if uploaded_file is not None:
        from components.patch_selector import PatchSelector
        from components.image_processor import ImageProcessor
        from components.results_viewer import ResultsViewer
        
        # Inicializar componentes
        patch_selector = PatchSelector(api_client)
        image_processor = ImageProcessor(api_client)
        results_viewer = ResultsViewer()
        
        # Cargar y mostrar imagen
        image = patch_selector.load_and_display_image(uploaded_file)
        
        if image is not None:
            # Layout principal
            col_canvas, col_config = st.columns([2, 1])
            
            with col_config:
                st.markdown('<h3 class="sub-header">⚙️ Configuración</h3>', unsafe_allow_html=True)
                
                # Selección de arquitectura
                architectures = list(set([model["architecture"] for model in available_models if model["available"]]))
                if not architectures:
                    st.error("No hay modelos disponibles")
                    return
                
                architecture = st.selectbox(
                    "🏗️ Arquitectura:",
                    architectures,
                    help="Selecciona la arquitectura de red neuronal",
                    key="func1_arch"
                )
                
                # Configuración de factor de escala
                config = patch_selector.show_scale_configuration(available_models, architecture)
                
                if config is None:
                    return
                
                # Opciones avanzadas
                with st.expander("🔬 Opciones Avanzadas"):
                    # Verificar estado de KimiaNet
                    kimianet_status = api_client.get_kimianet_status()
                    kimianet_available = kimianet_status and kimianet_status.get("available", False)
                    
                    if kimianet_available:
                        st.success("✅ KimiaNet disponible para evaluación perceptual")
                        evaluate_quality = st.checkbox(
                            "🧠 Evaluar calidad con KimiaNet",
                            value=True,
                            help="Calcula PSNR, SSIM e índice perceptual usando KimiaNet",
                            key="func1_quality"
                        )
                    else:
                        st.warning("⚠️ KimiaNet no disponible - solo PSNR/SSIM")
                        evaluate_quality = st.checkbox(
                            "📊 Evaluar calidad básica",
                            value=True,
                            help="Calcula PSNR y SSIM (KimiaNet no disponible)",
                            key="func1_quality_basic"
                        )
                
                # Botón de procesamiento
                process_button = st.button("🚀 Procesar Parche", type="primary", key="func1_process")
            
            with col_canvas:
                st.markdown('<h3 class="sub-header">🎯 Selección de Parche</h3>', unsafe_allow_html=True)
                
                # Canvas interactivo
                canvas_result = patch_selector.show_interactive_canvas(image, config["patch_size"])
                
                # Procesar si hay selección y se hace clic
                if canvas_result and process_button:
                    selection = patch_selector.extract_selection_coordinates(canvas_result, image, config["patch_size"])
                    
                    if selection:
                        # Procesar con upsampling por escala
                        with st.spinner("🔄 Procesando parche..."):
                            result = image_processor.process_by_scale(
                                uploaded_file, 
                                selection, 
                                config,
                                evaluate_quality=config.get("evaluate_quality", evaluate_quality)
                            )
                            
                            if result:
                                # Mostrar resultados
                                st.markdown("---")
                                results_viewer.display_scale_results(result)

def handle_functionality_2(api_client, available_models):
    """Maneja la Funcionalidad 2: Imagen Completa"""
    st.markdown('<h2 class="sub-header">🖼️ Procesamiento de Imagen Completa</h2>', unsafe_allow_html=True)
    
    st.info("🚧 **Funcionalidad en desarrollo**")
    st.markdown("""
    Esta funcionalidad incluirá:
    - Procesamiento automático por parches
    - Estrategias de división inteligente  
    - Manejo de overlap para evitar artefactos
    - Procesamiento en lotes
    - Selección automática de la mejor estrategia
    """)
    
    # Mostrar estado de modelos en sidebar
    show_model_status_sidebar(available_models)
    
    # Panel de configuración placeholder
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Configuración:**")
        target_resolution = st.selectbox(
            "Resolución objetivo:",
            ["2048x2048", "4096x4096", "8192x8192"],
            help="Selecciona la resolución final deseada",
            key="func2_resolution"
        )
        
        processing_strategy = st.radio(
            "Estrategia de procesamiento:",
            ["Automática (recomendada)", "Manual"],
            help="Automática: el sistema elige la mejor ruta",
            key="func2_strategy"
        )
    
    with col2:
        st.markdown("**Estado:**")
        st.markdown("- 🚧 División automática por parches")
        st.markdown("- 🚧 Manejo de overlapping")
        st.markdown("- 🚧 Reconstrucción inteligente")
        st.markdown("- ✅ Múltiples arquitecturas disponibles")
        
        if st.button("🚀 Procesar Imagen Completa", type="primary", key="func2_process"):
            st.warning("⏳ Funcionalidad próximamente disponible")

def handle_functionality_3(api_client, available_models):
    """Maneja la Funcionalidad 3: Evaluación Comparativa"""
    st.markdown('<h2 class="sub-header">📊 Evaluación Comparativa de Modelos</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Métricas disponibles:**
    - 📈 **PSNR** (Peak Signal-to-Noise Ratio)
    - 🔍 **SSIM** (Structural Similarity Index)
    - 🧠 **Índice Perceptual KimiaNet** (especializado en histopatología)
    - 📊 **MSE** (Mean Squared Error)
    - 🎨 **Evaluación Cualitativa** (mapas de diferencias)
    """)
    
    # Mostrar estado de modelos en sidebar
    show_model_status_sidebar(available_models)
    
    # Configuración de evaluación
    eval_mode = st.radio(
        "Modo de evaluación:",
        ["📝 Evaluar resultado único", "⚖️ Comparar múltiples arquitecturas"],
        key="func3_mode"
    )
    
    if eval_mode == "📝 Evaluar resultado único":
        handle_single_evaluation(api_client)
    else:
        handle_comparative_evaluation(api_client, available_models)

def handle_single_evaluation(api_client):
    """Maneja evaluación de un solo resultado"""
    st.markdown("### 📝 Evaluación de Resultado Único")
    
    col1, col2 = st.columns(2)
    
    with col1:
        original_file = st.file_uploader(
            "📁 Imagen Original (Baja Resolución):",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="func3_original"
        )
    
    with col2:
        enhanced_file = st.file_uploader(
            "📁 Imagen Mejorada (Alta Resolución):",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="func3_enhanced"
        )
    
    if original_file and enhanced_file:
        # Opciones de evaluación
        with st.expander("⚙️ Opciones de Evaluación"):
            calculate_perceptual = st.checkbox(
                "🧠 Calcular índice perceptual (KimiaNet)",
                value=True,
                help="Más lento pero más preciso para histopatología"
            )
            
            show_difference_map = st.checkbox(
                "🎨 Mostrar mapa de diferencias",
                value=True,
                help="Visualización cualitativa de diferencias"
            )
        
        if st.button("📊 Evaluar Calidad", type="primary", key="func3_eval_single"):
            with st.spinner("🔄 Evaluando calidad..."):
                result = api_client.evaluate_image_quality(
                    original_file, 
                    enhanced_file,
                    calculate_perceptual=calculate_perceptual
                )
                
                if result:
                    from components.results_viewer import ResultsViewer
                    results_viewer = ResultsViewer()
                    results_viewer.display_evaluation_results(result, show_difference_map)

def handle_comparative_evaluation(api_client, available_models):
    """Maneja evaluación comparativa entre múltiples arquitecturas"""
    st.markdown("### ⚖️ Comparación de Múltiples Arquitecturas")
    st.info("🚧 **Funcionalidad en desarrollo**")
    
    st.markdown("""
    Esta funcionalidad permitirá:
    - Procesar la misma imagen con múltiples arquitecturas
    - Comparar métricas lado a lado
    - Generar reportes comparativos
    - Análisis estadístico de rendimiento
    """)

if __name__ == "__main__":
    main()