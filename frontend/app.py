# frontend/app.py - Versión completamente renovada

#!/usr/bin/env python3
"""
Frontend Streamlit MEJORADO para aplicación de superresolución
Funcionalidad 1: Selección y procesamiento inteligente de parches
"""

import streamlit as st
import requests
import numpy as np
from PIL import Image
import base64
import io
import cv2
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="🔬 Microscopy Super-Resolution - Parches Inteligentes",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS mejorados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #E6F3FF;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4682B4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .success-card {
        background: linear-gradient(135deg, #f0fff0 0%, #e8f5e8 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fffaf0 0%, #fff2e6 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FFA500;
        margin: 0.5rem 0;
    }
    
    .comparison-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
    }
    
    .model-selector {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4682B4 0%, #2E8B57 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Estado de sesión mejorado
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = {
        'uploaded_image': None,
        'current_selection': None,
        'selected_model': None,
        'processing_results': None,
        'quality_metrics': None,
        'comparison_results': None
    }

if 'ui_state' not in st.session_state:
    st.session_state.ui_state = {
        'show_advanced': False,
        'auto_adjust_patch': True,
        'calculate_metrics': True,
        'compare_architectures': False
    }

class APIClient:
    """Cliente API mejorado con cache y manejo de errores"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.timeout = 30
    
    @st.cache_data(ttl=300)  # Cache 5 minutos
    def check_connection(_self):
        try:
            response = requests.get(f"{_self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @st.cache_data(ttl=300)
    def get_available_architectures(_self):
        try:
            response = requests.get(f"{_self.base_url}/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                architectures = list(set([m["architecture"] for m in models if m["available"]]))
                return sorted(architectures)
            return []
        except:
            return []
    
    @st.cache_data(ttl=300)
    def get_specific_models(_self, architecture):
        try:
            response = requests.get(f"{_self.base_url}/models/specific/{architecture}", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def validate_patch_selection(self, x, y, width, height, img_width, img_height, target_size=None):
        try:
            params = {
                "x": x, "y": y, "width": width, "height": height,
                "image_width": img_width, "image_height": img_height
            }
            if target_size:
                params["target_size"] = target_size
            
            response = requests.get(f"{self.base_url}/patch/validate_selection", params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def preview_patch(self, image_file, x, y, width, height, target_size=None):
        try:
            files = {"file": ("image.png", image_file, "image/png")}
            data = {"x": x, "y": y, "width": width, "height": height}
            if target_size:
                data["target_size"] = target_size
            
            response = requests.post(f"{self.base_url}/patch/preview", files=files, data=data, timeout=15)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def process_patch_advanced(self, image_file, model_name, x, y, width, height, 
                              target_size=None, evaluate_quality=False, compare_architectures=False):
        try:
            files = {"file": ("image.png", image_file, "image/png")}
            data = {
                "model_name": model_name,
                "x": x, "y": y, "width": width, "height": height,
                "evaluate_quality": evaluate_quality,
                "compare_architectures": compare_architectures
            }
            if target_size:
                data["target_size"] = target_size
            
            response = requests.post(f"{self.base_url}/process_patch_advanced", 
                                   files=files, data=data, timeout=60)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error procesando: {e}")
            return None

def show_metric_card(title, value, interpretation="", color="blue"):
    """Muestra tarjeta de métrica mejorada"""
    color_map = {
        "blue": "#4682B4",
        "green": "#2E8B57", 
        "orange": "#FFA500",
        "red": "#DC143C"
    }
    
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 0.5rem; 
                border-left: 4px solid {color_map.get(color, '#4682B4')}; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem 0;">
        <h4 style="margin: 0; color: {color_map.get(color, '#4682B4')};">{title}</h4>
        <h2 style="margin: 0.25rem 0; color: #333;">{value}</h2>
        {f'<p style="margin: 0; color: #666; font-size: 0.9rem;">{interpretation}</p>' if interpretation else ''}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🔬 Microscopy Super-Resolution</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">📐 Funcionalidad 1: Selección Inteligente de Parches</h2>', unsafe_allow_html=True)
    
    # Inicializar cliente API
    api_client = APIClient()
    
    # Verificar conexión
    if not api_client.check_connection():
        st.error("❌ **No se puede conectar con la API**")
        st.info("🔧 **Solución:** Ejecuta `cd backend && python main.py` en otra terminal")
        st.stop()
    
    # Sidebar con configuración
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Configuración</h2>', unsafe_allow_html=True)
        
        # Selección de arquitectura
        architectures = api_client.get_available_architectures()
        if not architectures:
            st.error("No hay arquitecturas disponibles")
            st.stop()
        
        selected_architecture = st.selectbox(
            "🏗️ Arquitectura:",
            architectures,
            help="Selecciona la arquitectura de red neuronal"
        )
        
        # Obtener modelos específicos
        models_data = api_client.get_specific_models(selected_architecture)
        if not models_data or not models_data["models"]:
            st.error(f"No hay modelos disponibles para {selected_architecture}")
            st.stop()
        
        available_models = [m for m in models_data["models"] if m["available"]]
        if not available_models:
            st.error(f"No hay modelos cargados para {selected_architecture}")
            st.stop()
        
        # Selección de modelo específico
        st.markdown("### 🎯 Modelo Específico")
        selected_model_idx = st.selectbox(
            "Modelo:",
            range(len(available_models)),
            format_func=lambda x: f"{available_models[x]['display_name']} {'✅' if available_models[x]['available'] else '❌'}",
            help="Selecciona el modelo específico a usar"
        )
        
        selected_model = available_models[selected_model_idx]
        st.session_state.processing_state['selected_model'] = selected_model
        
        # Mostrar información del modelo
        with st.expander("ℹ️ Información del Modelo"):
            st.write(f"**Entrada:** {selected_model['input_size']}×{selected_model['input_size']}px")
            st.write(f"**Salida:** {selected_model['output_size']}×{selected_model['output_size']}px")
            st.write(f"**Escala:** ×{selected_model['scale']}")
            if selected_model.get('checkpoint_iter'):
                st.write(f"**Checkpoint:** {selected_model['checkpoint_iter']}")
            if selected_model.get('discriminator'):
                st.write(f"**Discriminador:** {selected_model['discriminator']}")
        
        st.markdown("---")
        
        # Opciones avanzadas
        st.markdown("### 🔬 Opciones Avanzadas")
        st.session_state.ui_state['auto_adjust_patch'] = st.checkbox(
            "🎯 Ajuste automático de parche", 
            value=st.session_state.ui_state['auto_adjust_patch'],
            help="Ajusta automáticamente el parche al tamaño del modelo"
        )
        
        st.session_state.ui_state['calculate_metrics'] = st.checkbox(
            "📊 Calcular métricas de calidad", 
            value=st.session_state.ui_state['calculate_metrics'],
            help="Incluye PSNR, SSIM, MSE e índice perceptual"
        )
        
        st.session_state.ui_state['compare_architectures'] = st.checkbox(
            "⚖️ Comparar arquitecturas", 
            value=st.session_state.ui_state['compare_architectures'],
            help="Compara con otras arquitecturas disponibles"
        )
    
    # Área principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">📁 Cargar Imagen</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Selecciona imagen de microscopía:",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Formatos: PNG, JPG, JPEG, TIFF, BMP"
        )
        
        if uploaded_file is not None:
            # Cargar y mostrar imagen
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            st.session_state.processing_state['uploaded_image'] = {
                'data': image_np,
                'file': uploaded_file,
                'shape': image_np.shape
            }
            
            # Información de la imagen
            st.markdown("#### 📊 Información de la Imagen")
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                show_metric_card("Ancho", f"{image_np.shape[1]}px", color="blue")
            with info_col2:
                show_metric_card("Alto", f"{image_np.shape[0]}px", color="blue")
            with info_col3:
                channels = image_np.shape[2] if len(image_np.shape) == 3 else 1
                show_metric_card("Canales", str(channels), color="blue")
            
            # Canvas interactivo
            st.markdown("#### 🎯 Selección de Parche")
            
            # Calcular dimensiones del canvas
            max_width = 600
            max_height = 400
            img_height, img_width = image_np.shape[:2]
            
            scale_factor = min(max_width / img_width, max_height / img_height, 1.0)
            canvas_width = int(img_width * scale_factor)
            canvas_height = int(img_height * scale_factor)
            
            # Redimensionar imagen para canvas
            display_image = cv2.resize(image_np, (canvas_width, canvas_height))
            pil_image = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            
            # Canvas
            canvas_result = st_canvas(
                fill_color="rgba(70, 130, 180, 0.2)",
                stroke_width=3,
                stroke_color="#4682B4",
                background_image=pil_image,
                update_streamlit=True,
                width=canvas_width,
                height=canvas_height,
                drawing_mode="rect",
                key="patch_selector",
                display_toolbar=True
            )
            
            # Procesar selección
            if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
                rect = canvas_result.json_data["objects"][-1]
                
                # Coordenadas en imagen original
                x = int(rect["left"] / scale_factor)
                y = int(rect["top"] / scale_factor)
                width = int(rect["width"] / scale_factor)
                height = int(rect["height"] / scale_factor)
                
                # Validar selección
                target_size = selected_model['input_size'] if st.session_state.ui_state['auto_adjust_patch'] else None
                validation = api_client.validate_patch_selection(
                    x, y, width, height, img_width, img_height, target_size
                )
                
                if validation and validation["can_process"]:
                    selection = validation["selection"]
                    st.session_state.processing_state['current_selection'] = selection
                    
                    # Mostrar información de selección
                    st.markdown("#### ✅ Selección Validada")
                    
                    sel_col1, sel_col2 = st.columns(2)
                    with sel_col1:
                        st.write(f"**📍 Posición:** ({selection['x']}, {selection['y']})")
                        st.write(f"**📏 Tamaño:** {selection['width']}×{selection['height']}px")
                    
                    with sel_col2:
                        if selection.get('auto_adjusted'):
                            st.success("🎯 Ajustado automáticamente")
                        st.write(f"**🎯 Tamaño modelo:** {selected_model['input_size']}px")
                    
                    # Preview del parche
                    uploaded_file.seek(0)
                    preview_result = api_client.preview_patch(
                        uploaded_file, 
                        selection['x'], selection['y'], 
                        selection['width'], selection['height'],
                        selected_model['input_size'] if st.session_state.ui_state['auto_adjust_patch'] else None
                    )
                    
                    if preview_result and preview_result["success"]:
                        patch_img = base64.b64decode(preview_result["patch_preview"])
                        patch_pil = Image.open(io.BytesIO(patch_img))
                        
                        st.markdown("#### 👁️ Preview del Parche")
                        st.image(patch_pil, caption=f"Parche {preview_result['patch_size']}", width=200)
    
    with col2:
        st.markdown('<h3 class="sub-header">🚀 Procesamiento</h3>', unsafe_allow_html=True)
        
        if (st.session_state.processing_state['uploaded_image'] is not None and 
            st.session_state.processing_state['current_selection'] is not None):
            
            # Botón de procesamiento
            if st.button("🔬 Procesar Parche", type="primary", use_container_width=True):
                with st.spinner("🔄 Procesando con IA..."):
                    uploaded_file.seek(0)
                    selection = st.session_state.processing_state['current_selection']
                    
                    result = api_client.process_patch_advanced(
                        uploaded_file,
                        selected_model['name'],
                        selection['x'], selection['y'],
                        selection['width'], selection['height'],
                        selected_model['input_size'] if st.session_state.ui_state['auto_adjust_patch'] else None,
                        st.session_state.ui_state['calculate_metrics'],
                        st.session_state.ui_state['compare_architectures']
                    )
                    
                    if result and result["success"]:
                        st.session_state.processing_state['processing_results'] = result
                        st.session_state.processing_state['quality_metrics'] = result.get('quality_metrics')
                        st.success("✅ ¡Procesamiento completado!")
                    else:
                        st.error("❌ Error en el procesamiento")
            
            # Mostrar resultados si existen
            if st.session_state.processing_state['processing_results']:
                show_processing_results()
        else:
            st.info("📋 **Instrucciones:**\n1. Carga una imagen\n2. Selecciona un parche\n3. Haz clic en 'Procesar Parche'")

def show_processing_results():
    """Muestra resultados de procesamiento de forma visual y atractiva"""
    results = st.session_state.processing_state['processing_results']
    
    st.markdown('<h3 class="sub-header">🎉 Resultados del Procesamiento</h3>', unsafe_allow_html=True)
    
    # Información del modelo usado
    model_info = results["model_info"]
    st.markdown(f"""
    <div class="success-card">
        <h4>🏗️ Modelo Utilizado</h4>
        <p><strong>Arquitectura:</strong> {model_info['architecture']}</p>
        <p><strong>Resolución:</strong> {model_info['input_size']}×{model_info['input_size']} → {model_info['output_size']}×{model_info['output_size']}</p>
        <p><strong>Factor de Escala:</strong> ×{model_info['scale_factor']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparación visual
    st.markdown("#### 🔍 Comparación Visual Real")
    
    # Decodificar imágenes
    original_img = base64.b64decode(results["original_patch"])
    input_img = base64.b64decode(results["input_patch"])
    enhanced_img = base64.b64decode(results["enhanced_patch"])
    
    original_pil = Image.open(io.BytesIO(original_img))
    input_pil = Image.open(io.BytesIO(input_img))
    enhanced_pil = Image.open(io.BytesIO(enhanced_img))
    
    # Layout de comparación
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        st.markdown("**📷 Parche Original**")
        st.image(original_pil, caption=f"Tamaño: {results['sizes']['original']}", use_column_width=True)
    
    with comp_col2:
        st.markdown("**📥 Entrada Modelo**")
        st.image(input_pil, caption=f"Tamaño: {results['sizes']['input']}", use_column_width=True)
    
    with comp_col3:
        st.markdown("**✨ Resultado SR**")
        st.image(enhanced_pil, caption=f"Tamaño: {results['sizes']['enhanced']}", use_column_width=True)
    
    # Métricas de calidad
    if results.get("quality_metrics") and "error" not in results["quality_metrics"]:
        show_quality_metrics(results["quality_metrics"])
    
    # Comparación con otras arquitecturas
    if results.get("architecture_comparison"):
        show_architecture_comparison(results["architecture_comparison"], enhanced_pil)
    
    # Opciones de descarga
    st.markdown("#### 💾 Descargar Resultados")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        enhanced_bytes = io.BytesIO()
        enhanced_pil.save(enhanced_bytes, format='PNG')
        st.download_button(
            label="📥 Descargar Imagen Mejorada",
            data=enhanced_bytes.getvalue(),
            file_name=f"enhanced_{model_info['architecture'].lower()}_x{model_info['scale_factor']}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with download_col2:
        if st.button("📊 Exportar Reporte", use_container_width=True):
            st.info("🚧 Función de reporte en desarrollo")

def show_quality_metrics(metrics):
    """Muestra métricas de calidad de forma visual"""
    st.markdown("#### 📈 Métricas de Calidad")
    
    if "error" in metrics:
        st.error(f"Error en métricas: {metrics['error']}")
        return
    
    # Métricas numéricas
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        psnr = metrics.get("psnr", -1)
        if psnr > 0:
            color = "green" if psnr > 30 else "orange" if psnr > 25 else "red"
            show_metric_card("PSNR", f"{psnr:.2f} dB", 
                           metrics.get("interpretation", {}).get("psnr", ""), color)
    
    with metric_col2:
        ssim = metrics.get("ssim", -1)
        if ssim > 0:
            color = "green" if ssim > 0.9 else "orange" if ssim > 0.8 else "red"
            show_metric_card("SSIM", f"{ssim:.4f}", 
                           metrics.get("interpretation", {}).get("ssim", ""), color)
    
    with metric_col3:
        mse = metrics.get("mse", -1)
        if mse > 0:
            color = "green" if mse < 500 else "orange" if mse < 1000 else "red"
            show_metric_card("MSE", f"{mse:.1f}", 
                           metrics.get("interpretation", {}).get("mse", ""), color)
    
    with metric_col4:
        perceptual = metrics.get("perceptual_index", -1)
        if perceptual >= 0:
            color = "green" if perceptual < 0.01 else "orange" if perceptual < 0.1 else "red"
            show_metric_card("Índice Perceptual", f"{perceptual:.6f}", 
                           metrics.get("interpretation", {}).get("perceptual", ""), color)
    
    # Gráfico de radar para métricas normalizadas
    if all(metrics.get(m, -1) >= 0 for m in ["psnr", "ssim"]):
        show_metrics_radar_chart(metrics)
    
    # Información sobre KimiaNet
    if metrics.get("kimianet_used"):
        st.success("🧠 **KimiaNet Utilizado:** Índice perceptual especializado en histopatología")
    else:
        st.warning("⚠️ **KimiaNet No Disponible:** Solo métricas básicas calculadas")

def show_metrics_radar_chart(metrics):
    """Muestra gráfico de radar para métricas"""
    try:
        # Normalizar métricas para el radar (0-1)
        psnr_norm = min(metrics.get("psnr", 0) / 40, 1.0)  # Max 40 dB
        ssim_norm = metrics.get("ssim", 0)  # Ya está 0-1
        mse_norm = max(0, 1 - (metrics.get("mse", 1000) / 2000))  # Invertir MSE
        perceptual_norm = max(0, 1 - (metrics.get("perceptual_index", 0.1) / 0.2))  # Invertir perceptual
        
        # Crear gráfico
        categories = ['PSNR', 'SSIM', 'MSE (inv)', 'Perceptual (inv)']
        values = [psnr_norm, ssim_norm, mse_norm, perceptual_norm]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Métricas de Calidad',
            line_color='#4682B4',
            fillcolor='rgba(70, 130, 180, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Perfil de Calidad (Normalizado)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creando gráfico: {e}")

def show_architecture_comparison(comparisons, reference_img):
    """Muestra comparación entre arquitecturas"""
    if not comparisons:
        return
    
    st.markdown("#### ⚖️ Comparación entre Arquitecturas")
    
    # Crear columnas para comparación
    cols = st.columns(len(comparisons) + 1)
    
    # Imagen de referencia
    with cols[0]:
        st.markdown("**📘 Referencia**")
        st.image(reference_img, caption="Resultado principal", use_column_width=True)
    
    # Resultados alternativos
    for i, comp in enumerate(comparisons):
        with cols[i + 1]:
            st.markdown(f"**🏗️ {comp['architecture']}**")
            alt_img = base64.b64decode(comp["result"])
            alt_pil = Image.open(io.BytesIO(alt_img))
            st.image(alt_pil, caption=comp["architecture"], use_column_width=True)

if __name__ == "__main__":
    main()