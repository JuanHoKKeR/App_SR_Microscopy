"""
Configuración de UI y estilos CSS para Streamlit actualizada
"""

import streamlit as st

def setup_page_config():
    """Configura la página de Streamlit"""
    st.set_page_config(
        page_title="Microscopy Super-Resolution",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_custom_css():
    """Carga CSS personalizado"""
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

    .info-box {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #4682B4;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .success-box {
        background: linear-gradient(135deg, #f0fff0 0%, #e8f5e8 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #2E8B57;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .error-box {
        background: linear-gradient(135deg, #fff0f0 0%, #ffe8e8 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #DC143C;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .warning-box {
        background: linear-gradient(135deg, #fffaf0 0%, #fff2e6 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #FFA500;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .metric-container {
        background: white;
        padding: 1.2rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E8B57;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.25rem;
    }

    .progress-step {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }

    .progress-step.active {
        background: #e3f2fd;
        border-color: #2196f3;
        box-shadow: 0 2px 4px rgba(33, 150, 243, 0.3);
    }

    .progress-step.completed {
        background: #e8f5e8;
        border-color: #4caf50;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3);
    }

    .canvas-container {
        border: 2px solid #ddd;
        border-radius: 0.5rem;
        padding: 0.5rem;
        background: white;
    }

    .instruction-text {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
        margin: 0.5rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        border-left: 3px solid #4682B4;
    }

    .model-status-available {
        color: #2E8B57;
        font-weight: bold;
    }

    .model-status-unavailable {
        color: #DC143C;
        font-weight: bold;
    }

    .comparison-container {
        display: flex;
        justify-content: space-around;
        align-items: flex-start;
        gap: 1rem;
        margin: 1rem 0;
    }

    .comparison-item {
        flex: 1;
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .arrow-right {
        font-size: 2rem;
        color: #4682B4;
        align-self: center;
    }

    /* Tabs personalizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px;
        gap: 8px;
        padding-left: 12px;
        padding-right: 12px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4682B4;
        color: white;
    }

    /* Animaciones */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .processing {
        animation: pulse 1.5s infinite;
    }

    /* Responsivo */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .comparison-container {
            flex-direction: column;
        }
        
        .arrow-right {
            transform: rotate(90deg);
        }
    }

    /* Mejorar file uploader */
    .stFileUploader > div {
        border: 2px dashed #4682B4;
        border-radius: 0.75rem;
        background: #f8f9fa;
        transition: all 0.3s ease;
        padding: 2rem;
    }

    .stFileUploader > div:hover {
        border-color: #2E8B57;
        background: #f0f8ff;
    }

    /* Mejoras para botones */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4682B4 0%, #2E8B57 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* Selectbox personalizado */
    .stSelectbox > div > div {
        background: white;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
    }

    /* Expander personalizado */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
    
    /* Mejorar las columnas proporcionales */
    .proportional-original {
        max-width: 300px;
    }
    
    .proportional-enhanced {
        flex: 2;
    }
    
    /* CSS para métricas mejoradas */
    .quality-metric-excellent {
        border-left: 4px solid #4CAF50;
        background: linear-gradient(135deg, #E8F5E8 0%, #F0FFF0 100%);
    }
    
    .quality-metric-good {
        border-left: 4px solid #FF9800;
        background: linear-gradient(135deg, #FFF3E0 0%, #FFFAF0 100%);
    }
    
    .quality-metric-poor {
        border-left: 4px solid #F44336;
        background: linear-gradient(135deg, #FFEBEE 0%, #FFF0F0 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def show_info_box(message: str, box_type: str = "info"):
    """Muestra una caja de información con estilo personalizado"""
    if box_type == "info":
        st.markdown(f'<div class="info-box">{message}</div>', unsafe_allow_html=True)
    elif box_type == "success":
        st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
    elif box_type == "error":
        st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)
    elif box_type == "warning":
        st.markdown(f'<div class="warning-box">{message}</div>', unsafe_allow_html=True)

def show_metric_card(value: str, label: str, quality_level: str = None):
    """Muestra una tarjeta de métrica personalizada"""
    quality_class = ""
    if quality_level:
        quality_class = f"quality-metric-{quality_level}"
    
    st.markdown(f"""
    <div class="metric-container {quality_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def show_progress_steps(steps: list, current_step: int = 0):
    """Muestra pasos de progreso visual"""
    st.markdown('<div style="margin: 1rem 0;">', unsafe_allow_html=True)
    
    for i, step in enumerate(steps):
        if i < current_step:
            class_name = "progress-step completed"
            icon = "✅"
        elif i == current_step:
            class_name = "progress-step active"
            icon = "🔄"
        else:
            class_name = "progress-step"
            icon = "⏳"
        
        st.markdown(f"""
        <div class="{class_name}">
            {icon} <strong>Paso {i+1}:</strong> {step}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_comparison_layout(original_title: str, enhanced_title: str, 
                         original_image, enhanced_image, scale_factor: int = 2):
    """Layout de comparación de imágenes con tamaños proporcionales"""
    st.markdown(f"""
    <div class="comparison-container">
        <div class="comparison-item">
            <h4>{original_title}</h4>
        </div>
        <div class="arrow-right">→</div>
        <div class="comparison-item">
            <h4>{enhanced_title}</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Crear columnas proporcionales al factor de escala
    col_widths = [1, scale_factor] if scale_factor > 1 else [1, 1]
    col1, col2 = st.columns(col_widths)
    
    with col1:
        st.image(original_image, use_column_width=True)
    with col2:
        st.image(enhanced_image, use_column_width=True)

def show_architecture_info(architecture: str):
    """Muestra información específica de cada arquitectura"""
    architecture_info = {
        "ESRGAN": {
            "description": "Enhanced Super-Resolution GAN especializada en histopatología",
            "strengths": ["Detalles muy finos", "Texturas realistas", "Discriminador KimiaNet"],
            "considerations": ["Tiempo de procesamiento mayor", "Puede generar artefactos en algunos casos"],
            "best_for": "Imágenes que requieren alta fidelidad de textura"
        },
        "SwinIR": {
            "description": "Swin Transformer para restauración de imágenes",
            "strengths": ["Balance calidad/velocidad", "Preserva estructuras", "Eficiente en memoria"],
            "considerations": ["Menos detalles finos que ESRGAN", "Requiere tamaños específicos"],
            "best_for": "Procesamiento rápido con buena calidad"
        },
        "EDSR": {
            "description": "Enhanced Deep Super-Resolution con arquitectura residual",
            "strengths": ["Muy eficiente", "Resultados suaves", "Estable"],
            "considerations": ["Puede ser menos detallado", "Tendencia a suavizar"],
            "best_for": "Procesamiento rápido y estable"
        }
    }
    
    if architecture in architecture_info:
        info = architecture_info[architecture]
        
        st.markdown(f"**📝 {info['description']}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Fortalezas:**")
            for strength in info['strengths']:
                st.markdown(f"- {strength}")
        
        with col2:
            st.markdown("**⚠️ Consideraciones:**")
            for consideration in info['considerations']:
                st.markdown(f"- {consideration}")
        
        st.markdown(f"**🎯 Mejor para:** {info['best_for']}")

def show_processing_status(status: str, details: str = ""):
    """Muestra estado de procesamiento"""
    status_icons = {
        "preparing": "📁",
        "processing": "🔄",
        "finalizing": "✨",
        "completed": "✅",
        "error": "❌"
    }
    
    icon = status_icons.get(status, "ℹ️")
    
    st.markdown(f"""
    <div class="info-box">
        {icon} <strong>{status.title()}</strong>
        {f"<br>{details}" if details else ""}
    </div>
    """, unsafe_allow_html=True)