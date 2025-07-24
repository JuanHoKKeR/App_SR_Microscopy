"""
Funcionalidad 1: Selección de Parches y Super-Resolución
Permite seleccionar parches de una imagen y aplicar super-resolución con diferentes arquitecturas
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import base64
import io
import time

from components.ui_config import show_info_box, show_metric_card, show_progress_steps, show_comparison_layout
from components.api_client import APIClient

def funcionalidad_1_page(api_client: APIClient):
    """Página principal de la funcionalidad 1"""
    
    st.markdown('<h2 class="sub-header">🎯 Funcionalidad 1: Selección de Parches</h2>', unsafe_allow_html=True)
    
    show_info_box("""
    **Esta funcionalidad te permite:**
    - 📁 Cargar una imagen de microscopía
    - 🎯 Seleccionar un parche específico interactivamente
    - 🚀 Aplicar super-resolución con diferentes arquitecturas
    - 📊 Ver resultados con métricas de calidad (opcional)
    """)
    
    # Obtener modelos disponibles
    available_models = api_client.get_available_models()
    if not available_models:
        st.error("❌ No se pudieron cargar los modelos disponibles")
        return
    
    # Paso 1: Cargar imagen
    uploaded_file = upload_image_section()
    
    if uploaded_file is not None:
        # Paso 2: Mostrar información de imagen y configuración
        image = load_and_display_image(uploaded_file)
        
        if image is not None:
            # Configuración en columnas
            col_config, col_canvas = st.columns([1, 2])
            
            with col_config:
                # Paso 3: Configurar procesamiento
                config = configure_processing(available_models, api_client)
                
                if config is None:
                    return
                
                # Paso 4: Botón de procesamiento
                process_button = st.button("🚀 Procesar Parche", type="primary", use_container_width=True)
            
            with col_canvas:
                # Paso 5: Canvas interactivo para selección
                st.markdown("### 🎯 Seleccionar Parche")
                canvas_result = show_interactive_canvas(image)
            
            # Paso 6: Procesar si hay selección y se hace clic
            if canvas_result and process_button:
                selection = extract_selection_coordinates(canvas_result, image)
                
                if selection:
                    process_patch_workflow(api_client, uploaded_file, selection, config)

def upload_image_section():
    """Sección para cargar imagen"""
    st.markdown("### 📁 Cargar Imagen de Microscopía")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen:",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Formatos soportados: PNG, JPG, JPEG, TIFF, BMP"
    )
    
    return uploaded_file

def load_and_display_image(uploaded_file):
    """Carga y muestra información de la imagen"""
    try:
        # Cargar imagen
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convertir a BGR si es necesario
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Mostrar información
        st.markdown("### 📊 Información de la Imagen")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_metric_card(f"{image_np.shape[1]}", "Ancho (px)")
        with col2:
            show_metric_card(f"{image_np.shape[0]}", "Alto (px)")
        with col3:
            channels = image_np.shape[2] if len(image_np.shape) == 3 else 1
            show_metric_card(f"{channels}", "Canales")
        with col4:
            size_mb = (image_np.nbytes / (1024 * 1024))
            show_metric_card(f"{size_mb:.1f} MB", "Tamaño")
        
        return image_np
        
    except Exception as e:
        st.error(f"Error cargando imagen: {e}")
        return None

def configure_processing(available_models, api_client):
    """Configuración de procesamiento"""
    st.markdown("### ⚙️ Configuración")
    
    # Selección de arquitectura
    architectures = list(set([model["architecture"] for model in available_models if model["available"]]))
    
    if not architectures:
        st.error("No hay modelos disponibles")
        return None
    
    architecture = st.selectbox(
        "🏗️ Arquitectura:",
        architectures,
        help="Selecciona la arquitectura de red neuronal"
    )
    
    # Obtener recomendaciones para la arquitectura
    recommendations = api_client.get_model_recommendations(architecture)
    
    # Configuración de tamaño de parche
    available_sizes = recommendations.get("available_sizes", [64, 128, 256, 512])
    if not available_sizes:
        available_sizes = [256]
    
    recommended_size = recommendations.get("recommended_start", 256)
    default_index = available_sizes.index(recommended_size) if recommended_size in available_sizes else 0
    
    patch_size = st.selectbox(
        "📐 Tamaño del Parche:",
        available_sizes,
        index=default_index,
        help=f"Tamaños disponibles para {architecture}"
    )
    
    # Configuración de factor de escalado
    max_recommended_scale = min(recommendations.get("max_scale", 16), 16)
    scale_options = [2**i for i in range(1, int(np.log2(max_recommended_scale)) + 2) if 2**i <= 16]
    
    target_scale = st.selectbox(
        "🔍 Factor de Escalado:",
        scale_options,
        index=0,
        help=f"Factor de escalado (máximo recomendado: x{max_recommended_scale})"
    )
    
    # Validar factibilidad
    is_feasible, message = api_client.validate_upsampling_feasibility(
        architecture, patch_size, target_scale
    )
    
    if is_feasible:
        st.success(f"✅ {message}")
        
        # Mostrar ruta de procesamiento
        path_info = api_client.get_upsampling_path(architecture, patch_size, target_scale)
        if path_info:
            with st.expander("🛤️ Ver ruta de procesamiento"):
                st.markdown(f"**Pasos requeridos:** {len(path_info['path'])}")
                for i, model_name in enumerate(path_info['path']):
                    st.markdown(f"**{i+1}.** {model_name}")
    else:
        st.error(f"❌ {message}")
        return None
    
    # Opciones avanzadas
    with st.expander("🔬 Opciones Avanzadas"):
        # Verificar estado de KimiaNet
        kimianet_status = api_client.get_kimianet_status()
        kimianet_available = kimianet_status and kimianet_status.get("available", False)
        
        if kimianet_available:
            st.success("✅ KimiaNet disponible para evaluación perceptual")
            evaluate_quality = st.checkbox(
                "🧠 Evaluar calidad con KimiaNet",
                value=False,
                help="Calcula PSNR, SSIM e índice perceptual usando KimiaNet (toma más tiempo)"
            )
        else:
            st.warning("⚠️ KimiaNet no disponible - solo PSNR/SSIM")
            evaluate_quality = st.checkbox(
                "📊 Evaluar calidad básica",
                value=False,
                help="Calcula PSNR y SSIM (KimiaNet no disponible)"
            )
        
        # Información adicional
        st.markdown(f"""
        **ℹ️ Información del procesamiento:**
        - **Resolución de entrada:** {patch_size}×{patch_size}px
        - **Resolución de salida:** {patch_size * target_scale}×{patch_size * target_scale}px
        - **Tiempo estimado:** ~{len(path_info['path']) * 2.5:.1f} segundos
        """)
    
    return {
        "architecture": architecture,
        "patch_size": patch_size,
        "target_scale": target_scale,
        "path_info": path_info,
        "evaluate_quality": evaluate_quality
    }

def show_interactive_canvas(image):
    """Canvas interactivo para selección de parches"""
    
    # Calcular dimensiones del canvas
    max_canvas_width = 800
    max_canvas_height = 600
    
    img_height, img_width = image.shape[:2]
    
    width_scale = max_canvas_width / img_width
    height_scale = max_canvas_height / img_height
    scale_factor = min(width_scale, height_scale, 1.0)
    
    canvas_width = int(img_width * scale_factor)
    canvas_height = int(img_height * scale_factor)
    
    # Redimensionar imagen para mostrar
    if scale_factor < 1.0:
        display_image = cv2.resize(image, (canvas_width, canvas_height))
    else:
        display_image = image.copy()
    
    # Instrucciones
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
    🖱️ <strong>Instrucciones:</strong> Dibuja un rectángulo sobre la región que deseas procesar.
    El parche se ajustará automáticamente al tamaño configurado.
    </div>
    """, unsafe_allow_html=True)
    
    # Canvas interactivo
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.2)",
        stroke_width=3,
        stroke_color="#FF4500",
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
        show_selection_info(canvas_result, image.shape, scale_factor)
    else:
        show_info_box("Dibuja un rectángulo para seleccionar el parche a procesar", "info")
    
    return canvas_result

def show_selection_info(canvas_result, image_shape, scale_factor):
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
        
        # Advertencias
        if width != height:
            st.warning(f"⚠️ El parche se cuadrará automáticamente (se usará: {min(width, height)}px)")
        
        if width < 64 or height < 64:
            st.error("❌ El parche es muy pequeño (mínimo 64x64 px)")
            
    except Exception as e:
        st.error(f"Error mostrando información de selección: {e}")

def extract_selection_coordinates(canvas_result, image):
    """Extrae coordenadas de la selección del canvas"""
    if not canvas_result.json_data or not canvas_result.json_data["objects"]:
        st.warning("⚠️ No hay selección. Dibuja un rectángulo primero.")
        return None
    
    try:
        rect = canvas_result.json_data["objects"][-1]
        
        # Calcular factor de escala
        max_canvas_width = 800
        max_canvas_height = 600
        img_height, img_width = image.shape[:2]
        
        width_scale = max_canvas_width / img_width
        height_scale = max_canvas_height / img_height
        scale_factor = min(width_scale, height_scale, 1.0)
        
        # Coordenadas en la imagen original
        x = int(rect["left"] / scale_factor)
        y = int(rect["top"] / scale_factor)
        width = int(rect["width"] / scale_factor)
        height = int(rect["height"] / scale_factor)
        
        # Validar coordenadas
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

def process_patch_workflow(api_client, uploaded_file, selection, config):
    """Workflow completo de procesamiento de parches"""
    
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🔄 Procesamiento en Curso</h3>', unsafe_allow_html=True)
    
    # Contenedores para UI de progreso
    progress_container = st.container()
    results_container = st.container()
    
    # Preparar pasos para progreso
    steps = [f"Aplicar {model}" for model in config["path_info"]["path"]]
    steps.insert(0, "Preparando imagen")
    steps.append("Finalizando")
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        steps_display = st.empty()
        
        # Mostrar pasos iniciales
        with steps_display:
            show_progress_steps(steps, 0)
    
    try:
        # Paso 1: Preparación
        status_text.text("📁 Preparando imagen...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        with steps_display:
            show_progress_steps(steps, 1)
        
        # Paso 2: Enviar a procesamiento
        status_text.text("🚀 Enviando a procesamiento...")
        progress_bar.progress(20)
        
        # Resetear archivo
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        
        # Llamar a la API
        result = api_client.process_sequential_upsampling(
            image_file=uploaded_file,
            architecture=config["architecture"],
            start_size=config["patch_size"],
            target_scale=config["target_scale"],
            x=selection["x"],
            y=selection["y"],
            width=selection["width"],
            height=selection["height"],
            evaluate_quality=config["evaluate_quality"]
        )
        
        if not result:
            st.error("❌ Error en el procesamiento")
            return
        
        # Simular progreso por pasos
        total_steps = len(config["path_info"]["path"])
        for i in range(total_steps):
            progress = 20 + (70 * (i + 1) / total_steps)
            progress_bar.progress(int(progress))
            status_text.text(f"🔧 Procesando con {config['path_info']['path'][i]}...")
            
            with steps_display:
                show_progress_steps(steps, i + 2)
            
            time.sleep(0.3)
        
        # Finalizar
        progress_bar.progress(95)
        status_text.text("✨ Finalizando...")
        
        with steps_display:
            show_progress_steps(steps, len(steps))
        
        time.sleep(0.5)
        progress_bar.progress(100)
        status_text.text("✅ ¡Procesamiento completado!")
        
        time.sleep(1)
        
        # Limpiar progreso y mostrar resultados
        progress_container.empty()
        
        with results_container:
            display_results(result)
            
    except Exception as e:
        st.error(f"❌ Error durante el procesamiento: {e}")

def display_results(result):
    """Muestra los resultados del procesamiento"""
    
    st.markdown('<h2 class="sub-header">🎉 Resultados del Procesamiento</h2>', unsafe_allow_html=True)
    
    # Resumen de procesamiento
    st.markdown("### 📊 Resumen")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_metric_card(
            result.get("architecture", "N/A"),
            "Arquitectura"
        )
    
    with col2:
        show_metric_card(
            f"×{result.get('target_scale', 'N/A')}",
            "Factor de Escala"
        )
    
    with col3:
        show_metric_card(
            f"{len(result.get('steps', []))}",
            "Pasos Aplicados"
        )
    
    with col4:
        original_size = result.get("original_size", "N/A")
        final_size = result.get("final_size", "N/A")
        show_metric_card(
            f"{original_size}→{final_size}",
            "Resolución"
        )
    
    # Comparación final
    st.markdown("### 🔍 Comparación Final")
    
    original_b64 = result.get("original_patch")
    final_b64 = result.get("final_result")
    
    if original_b64 and final_b64:
        # Convertir de base64 a imágenes
        original_img = base64_to_image(original_b64)
        final_img = base64_to_image(final_b64)
        
        if original_img and final_img:
            show_comparison_layout(
                "Imagen Original",
                f"Resultado (x{result.get('target_scale', 'N/A')})",
                original_img,
                final_img
            )
    
    # Métricas de calidad
    show_quality_metrics(result)
    
    # Opciones de descarga
    show_download_options(result)

def show_quality_metrics(result):
    """Muestra métricas de calidad si están disponibles"""
    quality_metrics = result.get("quality_metrics")
    
    if quality_metrics and "error" not in quality_metrics:
        st.markdown("### 📈 Métricas de Calidad")
        
        metrics = quality_metrics.get("metrics") or quality_metrics
        interpretation = quality_metrics.get("interpretation", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            psnr_val = metrics.get("psnr", -1)
            if psnr_val > 0:
                show_metric_card(
                    f"{psnr_val:.2f} dB",
                    f"PSNR - {interpretation.get('psnr', 'N/A')}"
                )
        
        with col2:
            ssim_val = metrics.get("ssim", -1)
            if ssim_val > 0:
                show_metric_card(
                    f"{ssim_val:.4f}",
                    f"SSIM - {interpretation.get('ssim', 'N/A')}"
                )
        
        with col3:
            perceptual_val = metrics.get("perceptual_index", -1)
            if perceptual_val >= 0:
                show_metric_card(
                    f"{perceptual_val:.6f}",
                    f"Índice Perceptual - {interpretation.get('perceptual', 'N/A')}"
                )
        
        # Información sobre KimiaNet
        kimianet_used = quality_metrics.get("kimianet_used", False)
        if kimianet_used:
            show_info_box("""
            🧠 **KimiaNet Utilizado:** Las métricas perceptuales se calcularon usando DenseNet121 
            con pesos KimiaNet, específicamente entrenado para imágenes de histopatología.
            """, "success")
        else:
            show_info_box("""
            ⚠️ **KimiaNet No Disponible:** Las métricas PSNR y SSIM están disponibles, 
            pero el índice perceptual no se pudo calcular sin KimiaNet.
            """, "warning")

def show_download_options(result):
    """Opciones de descarga"""
    st.markdown("### 💾 Descargar Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "final_result" in result:
            final_bytes = base64_to_bytes(result["final_result"])
            if final_bytes:
                st.download_button(
                    label="📥 Resultado Final",
                    data=final_bytes,
                    file_name=f"enhanced_x{result.get('target_scale', 'N')}_{result.get('architecture', 'unknown').lower()}.png",
                    mime="image/png",
                    type="primary"
                )
    
    with col2:
        if "original_patch" in result:
            original_bytes = base64_to_bytes(result["original_patch"])
            if original_bytes:
                st.download_button(
                    label="📥 Parche Original",
                    data=original_bytes,
                    file_name="original_patch.png",
                    mime="image/png"
                )
    
    with col3:
        if st.button("📄 Generar Reporte"):
            st.info("🚧 Función de reporte en desarrollo")

# Funciones auxiliares
def base64_to_image(base64_str):
    """Convierte base64 a imagen PIL"""
    try:
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data))
        return image
    except Exception:
        return None

def base64_to_bytes(base64_str):
    """Convierte base64 a bytes"""
    try:
        return base64.b64decode(base64_str)
    except Exception:
        return None