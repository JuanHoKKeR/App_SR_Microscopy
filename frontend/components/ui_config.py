"""
Configuración de UI y estilos CSS para Streamlit
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
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E8B57;
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
    }

    .progress-step.active {
        background: #e3f2fd;
        border-color: #2196f3;
    }

    .progress-step.completed {
        background: #e8f5e8;
        border-color: #4caf50;
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
        align-items: center;
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

    /* Ocultar elementos de Streamlit */
    .css-1d391kg {
        padding-top: 1rem;
    }

    .css-k1vhr4 {
        margin-top: -75px;
    }

    /* Botones personalizados */
    .stButton > button {
        background: linear-gradient(135deg, #4682B4 0%, #2E8B57 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* Selectbox personalizado */
    .stSelectbox > div > div {
        background: white;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
    }

    /* File uploader personalizado */
    .stFileUploader > div {
        border: 2px dashed #4682B4;
        border-radius: 0.75rem;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        border-color: #2E8B57;
        background: #f0f8ff;
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

def show_metric_card(value: str, label: str):
    """Muestra una tarjeta de métrica personalizada"""
    st.markdown(f"""
    <div class="metric-container">
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
                         original_image, enhanced_image):
    """Layout de comparación de imágenes"""
    st.markdown("""
    <div class="comparison-container">
        <div class="comparison-item">
            <h4>""" + original_title + """</h4>
        </div>
        <div class="arrow-right">→</div>
        <div class="comparison-item">
            <h4>""" + enhanced_title + """</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, use_column_width=True)
    with col2:
        st.image(enhanced_image, use_column_width=True)