"""
Funcionalidad 3: Evaluación y Comparación de Modelos
Permite comparar diferentes modelos y métricas de calidad
"""

import streamlit as st
import numpy as np
import pandas as pd
from components.ui_config import show_info_box, show_metric_card, show_comparison_layout
from components.api_client import APIClient

def funcionalidad_3_page(api_client: APIClient):
    """Página de funcionalidad 3 - Evaluación y comparación"""
    
    st.markdown('<h2 class="sub-header">📊 Funcionalidad 3: Evaluación y Comparación</h2>', unsafe_allow_html=True)
    
    show_info_box("""
    **Esta funcionalidad te permite:**
    - 📊 Comparar diferentes arquitecturas de super-resolución
    - 🔍 Evaluar métricas de calidad especializadas
    - 📈 Análisis visual con mapas de diferencias
    - 📋 Generar reportes comparativos detallados
    - 🧠 Evaluación perceptual con KimiaNet
    """)
    
    # Obtener modelos disponibles
    available_models = api_client.get_available_models()
    if not available_models:
        st.error("❌ No se pudieron cargar los modelos disponibles")
        return
    
    # Verificar estado de KimiaNet
    kimianet_status = api_client.get_kimianet_status()
    kimianet_available = kimianet_status and kimianet_status.get("available", False)
    
    # Tabs para diferentes tipos de evaluación
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Evaluación Individual", 
        "⚖️ Comparación Múltiple", 
        "📈 Análisis Visual", 
        "📋 Reportes"
    ])
    
    with tab1:
        individual_evaluation_tab(api_client, available_models, kimianet_available)
    
    with tab2:
        multiple_comparison_tab(api_client, available_models)
    
    with tab3:
        visual_analysis_tab(api_client)
    
    with tab4:
        reports_tab()

def individual_evaluation_tab(api_client, available_models, kimianet_available):
    """Tab para evaluación individual"""
    st.markdown("### 🔍 Evaluación Individual de Modelos")
    
    show_info_box("""
    **Evalúa un modelo específico** cargando una imagen LR y su correspondiente HR, 
    o permitiendo que el sistema genere la predicción y compare con ground truth.
    """, "info")
    
    # Modo de evaluación
    eval_mode = st.radio(
        "🎯 Modo de Evaluación:",
        [
            "📁 Cargar LR + HR (tengo ambas imágenes)",
            "🎯 Cargar HR + Generar predicción (solo tengo imagen HR)"
        ],
        help="Selecciona según las imágenes que tengas disponibles"
    )
    
    if eval_mode.startswith("📁"):
        evaluation_with_both_images(api_client, available_models, kimianet_available)
    else:
        evaluation_with_prediction(api_client, available_models, kimianet_available)

def evaluation_with_both_images(api_client, available_models, kimianet_available):
    """Evaluación cuando se tienen ambas imágenes"""
    st.markdown("#### 📁 Cargar Imágenes LR y HR")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lr_file = st.file_uploader(
            "🔽 Imagen de Baja Resolución (LR):",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="lr_upload"
        )
        
        if lr_file:
            st.image(lr_file, caption="Imagen LR", use_column_width=True)
    
    with col2:
        hr_file = st.file_uploader(
            "🔼 Imagen de Alta Resolución (HR):",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="hr_upload"
        )
        
        if hr_file:
            st.image(hr_file, caption="Imagen HR", use_column_width=True)
    
    if lr_file and hr_file:
        # Configuración de evaluación
        st.markdown("#### ⚙️ Configuración de Evaluación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Selección de modelos a evaluar
            architectures = list(set([model["architecture"] for model in available_models if model["available"]]))
            
            selected_architectures = st.multiselect(
                "🏗️ Arquitecturas a Evaluar:",
                architectures,
                default=architectures[:1] if architectures else [],
                help="Puedes seleccionar múltiples arquitecturas para comparar"
            )
        
        with col2:
            # Métricas a calcular
            st.markdown("**📊 Métricas a Calcular:**")
            
            calculate_psnr = st.checkbox("📈 PSNR", value=True)
            calculate_ssim = st.checkbox("🔍 SSIM", value=True)
            calculate_msssim = st.checkbox("🎯 MS-SSIM", value=False, help="Solo para imágenes grandes")
            
            if kimianet_available:
                calculate_perceptual = st.checkbox("🧠 Índice Perceptual (KimiaNet)", value=True)
                st.success("✅ KimiaNet disponible")
            else:
                calculate_perceptual = st.checkbox("🧠 Índice Perceptual (KimiaNet)", value=False, disabled=True)
                st.warning("⚠️ KimiaNet no disponible")
        
        # Botón de evaluación
        if st.button("🔍 Evaluar Modelos", type="primary") and selected_architectures:
            # Placeholder para evaluación
            st.info("🚧 **Evaluación en desarrollo**")
            
            # Simular resultados
            show_evaluation_results_placeholder(selected_architectures)

def evaluation_with_prediction(api_client, available_models, kimianet_available):
    """Evaluación generando predicción desde HR"""
    st.markdown("#### 🎯 Generar Predicción y Evaluar")
    
    hr_file = st.file_uploader(
        "🔼 Imagen de Alta Resolución (Ground Truth):",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        key="hr_pred_upload"
    )
    
    if hr_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(hr_file, caption="Imagen HR (Ground Truth)", use_column_width=True)
        
        with col2:
            st.markdown("**⚙️ Configuración:**")
            
            # Configuración de degradación
            degradation_factor = st.selectbox(
                "📉 Factor de Degradación:",
                [2, 4, 8],
                help="Factor para generar imagen LR desde HR"
            )
            
            # Selección de arquitectura
            architectures = list(set([model["architecture"] for model in available_models if model["available"]]))
            architecture = st.selectbox("🏗️ Arquitectura:", architectures)
            
            # Botón de procesamiento
            if st.button("🎯 Generar y Evaluar", type="primary"):
                st.info("🚧 **Funcionalidad en desarrollo**")

def multiple_comparison_tab(api_client, available_models):
    """Tab para comparación múltiple"""
    st.markdown("### ⚖️ Comparación Múltiple de Arquitecturas")
    
    show_info_box("""
    **Compara múltiples arquitecturas** en el mismo conjunto de imágenes 
    para obtener análisis estadísticos y rankings de rendimiento.
    """, "info")
    
    # Configuración de comparación
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📁 Dataset de Evaluación:**")
        
        dataset_mode = st.radio(
            "Modo de Dataset:",
            ["📂 Cargar Carpeta de Imágenes", "📋 Cargar Lista de Pares LR/HR"],
            help="Selecciona cómo proporcionarás las imágenes de prueba"
        )
        
        # Placeholder para carga de dataset
        if dataset_mode.startswith("📂"):
            st.file_uploader(
                "Cargar carpeta comprimida (.zip):",
                type=['zip'],
                help="Zip con imágenes LR y HR en carpetas separadas"
            )
        else:
            st.file_uploader(
                "Cargar archivo CSV con rutas:",
                type=['csv'],
                help="CSV con columnas: lr_path, hr_path"
            )
    
    with col2:
        st.markdown("**🏗️ Arquitecturas a Comparar:**")
        
        architectures = list(set([model["architecture"] for model in available_models if model["available"]]))
        
        selected_architectures = st.multiselect(
            "Selecciona arquitecturas:",
            architectures,
            default=architectures if len(architectures) <= 3 else architectures[:3],
            help="Máximo 5 arquitecturas para comparación efectiva"
        )
        
        if len(selected_architectures) > 5:
            st.warning("⚠️ Muchas arquitecturas seleccionadas. Recomendado: máximo 5.")
    
    # Configuración de métricas
    st.markdown("**📊 Configuración de Métricas:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Métricas Básicas:**")
        metrics_basic = st.multiselect(
            "Seleccionar:",
            ["PSNR", "SSIM", "MSE"],
            default=["PSNR", "SSIM"]
        )
    
    with col2:
        st.markdown("**Métricas Avanzadas:**")
        metrics_advanced = st.multiselect(
            "Seleccionar:",
            ["MS-SSIM", "LPIPS", "FID"],
            default=["MS-SSIM"]
        )
    
    with col3:
        st.markdown("**Métricas Especializadas:**")
        metrics_specialized = st.multiselect(
            "Seleccionar:",
            ["Índice Perceptual (KimiaNet)", "NIQE", "BRISQUE"],
            default=["Índice Perceptual (KimiaNet)"]
        )
    
    # Botón de comparación
    if st.button("⚖️ Iniciar Comparación", type="primary") and selected_architectures:
        st.info("🚧 **Comparación múltiple en desarrollo**")
        
        # Placeholder para resultados de comparación
        show_comparison_results_placeholder(selected_architectures)

def visual_analysis_tab(api_client):
    """Tab para análisis visual"""
    st.markdown("### 📈 Análisis Visual de Diferencias")
    
    show_info_box("""
    **Análisis visual especializado** que incluye mapas de diferencias, 
    análisis de gradientes, y visualización de artefactos.
    """, "info")
    
    # Cargar imágenes para análisis
    col1, col2 = st.columns(2)
    
    with col1:
        original_file = st.file_uploader(
            "🔼 Imagen Original/Ground Truth:",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="visual_original"
        )
        
        if original_file:
            st.image(original_file, caption="Original", use_column_width=True)
    
    with col2:
        processed_file = st.file_uploader(
            "🎯 Imagen Procesada:",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="visual_processed"
        )
        
        if processed_file:
            st.image(processed_file, caption="Procesada", use_column_width=True)
    
    if original_file and processed_file:
        st.markdown("### 🔍 Tipos de Análisis Visual")
        
        analysis_types = st.multiselect(
            "Selecciona análisis a realizar:",
            [
                "📊 Mapa de Diferencias Absolutas",
                "🌈 Mapa de Diferencias con Colores",
                "📈 Análisis de Gradientes",
                "🔍 Detección de Artefactos",
                "📋 Histograma de Diferencias",
                "🎯 Análisis de Regiones de Interés"
            ],
            default=["📊 Mapa de Diferencias Absolutas", "🌈 Mapa de Diferencias con Colores"]
        )
        
        if st.button("📈 Generar Análisis Visual", type="primary"):
            st.info("🚧 **Análisis visual en desarrollo**")
            
            # Placeholder para análisis visual
            show_visual_analysis_placeholder()

def reports_tab():
    """Tab para generar reportes"""
    st.markdown("### 📋 Generación de Reportes")
    
    show_info_box("""
    **Genera reportes detallados** de tus evaluaciones con gráficos, 
    tablas comparativas y análisis estadísticos.
    """, "info")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Tipo de Reporte:**")
        
        report_type = st.selectbox(
            "Selecciona tipo:",
            [
                "📋 Reporte de Evaluación Individual",
                "⚖️ Reporte de Comparación Múltiple",
                "📈 Reporte de Análisis Visual",
                "📊 Reporte Completo (Todo incluido)"
            ]
        )
        
        # Configuración de reporte
        include_images = st.checkbox("🖼️ Incluir imágenes en reporte", value=True)
        include_charts = st.checkbox("📊 Incluir gráficos y tablas", value=True)
        include_raw_data = st.checkbox("📋 Incluir datos en bruto", value=False)
    
    with col2:
        st.markdown("**📁 Formato de Salida:**")
        
        output_format = st.selectbox(
            "Formato:",
            ["PDF", "HTML", "Word (.docx)", "Excel (.xlsx)"],
            help="PDF recomendado para reportes profesionales"
        )
        
        # Configuración adicional
        report_name = st.text_input(
            "📝 Nombre del reporte:",
            value="Evaluacion_SuperResolucion",
            help="Sin espacios ni caracteres especiales"
        )
        
        # Metadatos
        author = st.text_input("👤 Autor:", value="Usuario")
        institution = st.text_input("🏢 Institución:", value="")
    
    # Botón de generación
    if st.button("📋 Generar Reporte", type="primary"):
        st.info("🚧 **Generación de reportes en desarrollo**")
        
        st.markdown(f"""
        **Reporte configurado:**
        - Tipo: {report_type}
        - Formato: {output_format}
        - Nombre: {report_name}
        - Autor: {author}
        {f"- Institución: {institution}" if institution else ""}
        """)

# Funciones placeholder para mostrar resultados
def show_evaluation_results_placeholder(architectures):
    """Muestra resultados placeholder de evaluación"""
    st.markdown("### 📊 Resultados de Evaluación")
    
    # Generar datos ficticios
    results_data = []
    for arch in architectures:
        results_data.append({
            "Arquitectura": arch,
            "PSNR (dB)": np.random.uniform(25, 35),
            "SSIM": np.random.uniform(0.8, 0.95),
            "Índice Perceptual": np.random.uniform(0.01, 0.1),
            "Tiempo (s)": np.random.uniform(1, 10)
        })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)
    
    # Gráfico de barras
    st.bar_chart(df.set_index("Arquitectura")[["PSNR (dB)", "SSIM"]])

def show_comparison_results_placeholder(architectures):
    """Muestra resultados placeholder de comparación"""
    st.markdown("### ⚖️ Resultados de Comparación")
    
    # Tabla de ranking
    ranking_data = []
    for i, arch in enumerate(architectures):
        ranking_data.append({
            "Ranking": i + 1,
            "Arquitectura": arch,
            "Score Global": np.random.uniform(0.7, 0.95),
            "PSNR Promedio": np.random.uniform(25, 35),
            "SSIM Promedio": np.random.uniform(0.8, 0.95)
        })
    
    df_ranking = pd.DataFrame(ranking_data)
    st.dataframe(df_ranking, use_container_width=True)

def show_visual_analysis_placeholder():
    """Muestra análisis visual placeholder"""
    st.markdown("### 🔍 Análisis Visual Generado")
    
    # Placeholder para diferentes tipos de análisis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Mapa de Diferencias:**")
        st.info("Aquí se mostraría el mapa de diferencias absolutas")
    
    with col2:
        st.markdown("**📈 Análisis de Gradientes:**")
        st.info("Aquí se mostraría el análisis de gradientes")
    
    # Métricas del análisis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_metric_card("15.2%", "Píxeles con Diferencia > 10")
    
    with col2:
        show_metric_card("92.3", "Score de Similaridad")
    
    with col3:
        show_metric_card("3", "Artefactos Detectados")