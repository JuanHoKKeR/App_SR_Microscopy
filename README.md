# App SR Microscopy - Demo Application

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/status-development-yellow.svg)](https://github.com/JuanHoKKeR/App_SR_Microscopy)

Una aplicación web interactiva para demostrar técnicas de super-resolución en imágenes de histopatología. Esta **aplicación DEMO** permite a los usuarios cargar imágenes de microscopia y aplicar diferentes modelos de super-resolución entrenados (ESRGAN, SwinIR, EDSR) de forma intuitiva.

## 🎯 **Objetivo de la Aplicación**

Proporcionar una interfaz web amigable que permita a investigadores, médicos y usuarios en general experimentar con los modelos de super-resolución desarrollados en el proyecto, facilitando la evaluación práctica de los resultados sin necesidad de conocimientos técnicos avanzados.

## ✨ **Características Principales**

- **🔬 Interfaz Intuitiva**: Diseño simple para carga y procesamiento de imágenes
- **🤖 Múltiples Modelos**: Integración de ESRGAN, SwinIR y EDSR
- **⚡ Procesamiento en Tiempo Real**: Resultados inmediatos con visualización comparativa
- **📊 Métricas Automáticas**: Cálculo de PSNR, SSIM y otras métricas de calidad
- **💾 Descarga de Resultados**: Exportación de imágenes procesadas
- **🌐 Arquitectura Web**: Backend FastAPI + Frontend Streamlit

## 🏗️ **Arquitectura del Sistema**

```
App_SR_Microscopy/
├── backend/                    # API Backend (FastAPI)
│   ├── main.py                # Servidor principal
│   ├── models/                # Carga y gestión de modelos
│   ├── utils/                 # Utilidades de procesamiento
│   └── requirements.txt       # Dependencias backend
├── frontend/                  # Interfaz Usuario (Streamlit)  
│   ├── app.py                # Aplicación principal
│   ├── components/           # Componentes UI personalizados
│   ├── assets/               # Recursos estáticos
└── shared/                   # Resources compartidos
    ├── models/               # Modelos entrenados (.pth .pt)
    ├── sample_images/        # Imágenes de ejemplo
    └── config/               # Configuraciones
```

## 🚀 **Instalación y Configuración**

### **Prerequisitos**
- Python 3.9+
- GPU NVIDIA (recomendado para inferencia rápida)
- 8GB+ RAM
- Modelos entrenados (.pth and .pt files)

### **1. Clonar el Repositorio**
```bash
git clone https://github.com/JuanHoKKeR/App_SR_Microscopy.git
cd App_SR_Microscopy
```

### **2. Configurar Backend (FastAPI)**

#### Crear Entorno Virtual
```bash
# Crear entorno virtual para backend
python -m venv backend_env

# Activar entorno virtual
# Windows
backend_env\Scripts\activate
# Linux/Mac
source backend_env/bin/activate
```

#### Instalar Dependencias
```bash
# Navegar al directorio backend
cd backend

# Instalar dependencias
pip install -r requirements.txt

# Dependencias principales incluyen:
# fastapi
# uvicorn[standard] 
# torch
# torchvision
# pillow
# numpy
# opencv-python
```

#### Configurar Modelos
```bash
# Crear directorio para modelos (si no existe)
mkdir -p ../shared/models

# Copiar modelos entrenados
# Ejemplo de estructura:
# shared/models/
# ├── esrgan_512to1024.pth
# ├── swinir_512to1024.pt
# └── edsr_512to1024.pt
```

### **3. Configurar Frontend (Streamlit)**

#### Crear Entorno Virtual
```bash
# Crear entorno virtual para frontend (desde directorio raíz)
python -m venv frontend_env

# Activar entorno virtual
# Windows
frontend_env\Scripts\activate
# Linux/Mac
source frontend_env/bin/activate
```

#### Instalar Dependencias
```bash
# Navegar al directorio frontend
cd frontend

# Instalar dependencias
pip install -r requirements.txt

# Dependencias principales incluyen:
# streamlit
# requests
# pillow
# numpy
# pandas
# plotly
```

## 🔧 **Ejecutar la Aplicación**

### **Paso 1: Iniciar Backend**

```bash
# Activar entorno backend
source backend_env/bin/activate  # Linux/Mac
# backend_env\Scripts\activate   # Windows

# Navegar al directorio backend
cd backend

# Ejecutar servidor FastAPI
python main.py
```

El servidor backend estará disponible en: `http://localhost:8000`

**Endpoints API disponibles:**
- `GET /`: Información del API
- `POST /upscale`: Endpoint principal para super-resolución
- `GET /models`: Lista de modelos disponibles
- `GET /health`: Estado del servidor

### **Paso 2: Iniciar Frontend**

```bash
# En nueva terminal, activar entorno frontend
source frontend_env/bin/activate  # Linux/Mac
# frontend_env\Scripts\activate   # Windows

# Navegar al directorio frontend
cd frontend

# Ejecutar aplicación Streamlit
streamlit run app.py
```

La aplicación web estará disponible en: `http://localhost:8501`

## 🖥️ **Interfaz de Usuario**

### **Pantalla Principal**

La aplicación presenta una interfaz dividida en secciones:

#### **1. Carga de Imagen**
```python
# Componente de upload en Streamlit
uploaded_file = st.file_uploader(
    "Seleccionar imagen de microscopia",
    type=['png', 'jpg', 'jpeg', 'tiff'],
    help="Formatos soportados: PNG, JPG, JPEG, TIFF"
)
```

#### **2. Selección de Modelo**
```python
# Selector de modelo
model_choice = st.selectbox(
    "Seleccionar modelo de super-resolución:",
    options=["ESRGAN", "SwinIR", "EDSR"],
    help="Cada modelo tiene características distintas"
)
```

#### **3. Configuración de Parámetros**
```python
# Parámetros ajustables
scale_factor = st.slider("Factor de escala", 2, 4, 2)
output_format = st.radio("Formato de salida", ["PNG", "JPEG"])
```

### **Visualización de Resultados**

#### **Comparación Lado a Lado**
```python
# Layout de comparación
col1, col2 = st.columns(2)

with col1:
    st.subheader("Imagen Original")
    st.image(original_image, caption="Input LR")

with col2:
    st.subheader("Resultado Super-Resolución")  
    st.image(sr_result, caption="Output HR")
```

#### **Métricas de Calidad**
```python
# Display de métricas
metrics_container = st.container()
with metrics_container:
    col1, col2, col3 = st.columns(3)
    col1.metric("PSNR", f"{psnr:.2f} dB")
    col2.metric("SSIM", f"{ssim:.3f}")
    col3.metric("Tiempo", f"{processing_time:.1f}s")
```

## 🔧 **Configuración Backend (FastAPI)**

### **Estructura main.py**
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io

app = FastAPI(title="Super-Resolution API", version="1.0.0")

# Cargar modelos al inicio
models = {
    "esrgan": load_esrgan_model(),
    "swinir": load_swinir_model(), 
    "edsr": load_edsr_model()
}

@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    model: str = "esrgan",
    scale: int = 2
):
    """Aplicar super-resolución a imagen cargada"""
    
    # Procesar imagen
    image = Image.open(io.BytesIO(await file.read()))
    
    # Aplicar modelo seleccionado
    result = apply_super_resolution(image, models[model], scale)
    
    # Calcular métricas (si ground truth disponible)
    metrics = calculate_metrics(image, result)
    
    return {
        "success": True,
        "metrics": metrics,
        "result_image": encode_image_base64(result)
    }

@app.get("/models")
async def get_available_models():
    """Listar modelos disponibles"""
    return {
        "models": list(models.keys()),
        "default": "esrgan"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 🎨 **Configuración Frontend (Streamlit)**

### **Estructura app.py**
```python
import streamlit as st
import requests
import base64
from PIL import Image
import io

# Configuración de página
st.set_page_config(
    page_title="Super-Resolution Demo",
    page_icon="🔬",
    layout="wide"
)

# Título principal
st.title("🔬 Super-Resolution para Microscopia")
st.markdown("Demostración interactiva de técnicas de super-resolución")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selector de modelo
    model = st.selectbox(
        "Modelo de SR:",
        ["esrgan", "swinir", "edsr"]
    )
    
    # Factor de escala
    scale = st.slider("Factor de escala:", 2, 4, 2)

# Área principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Cargar Imagen")
    
    uploaded_file = st.file_uploader(
        "Seleccionar imagen:",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        # Mostrar imagen original
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen Original")
        
        # Botón procesar
        if st.button("🚀 Aplicar Super-Resolución"):
            with st.spinner("Procesando..."):
                # Llamada al backend
                response = call_backend_api(uploaded_file, model, scale)
                
                if response["success"]:
                    st.session_state.result = response
                    st.success("¡Procesamiento completado!")

with col2:
    st.subheader("📈 Resultados")
    
    if "result" in st.session_state:
        result = st.session_state.result
        
        # Mostrar imagen resultado
        result_image = decode_base64_image(result["result_image"])
        st.image(result_image, caption="Super-Resolución")
        
        # Mostrar métricas
        metrics = result["metrics"]
        st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
        st.metric("SSIM", f"{metrics['ssim']:.3f}")
        
        # Botón descarga
        st.download_button(
            "📥 Descargar Resultado",
            data=encode_image_for_download(result_image),
            file_name="super_resolution_result.png",
            mime="image/png"
        )

def call_backend_api(file, model, scale):
    """Llamar API backend para procesamiento"""
    files = {"file": file.getvalue()}
    data = {"model": model, "scale": scale}
    
    response = requests.post(
        "http://localhost:8000/upscale",
        files=files,
        data=data
    )
    
    return response.json()
```

## 🖼️ **Galería de Ejemplos**

### **Casos de Uso Demostrados**

#### **Ejemplo 1: Mejora de Resolución ×2**
```
Input:  256×256 → Output: 512×512
Modelo: ESRGAN
PSNR:   28.5 dB
SSIM:   0.87
Tiempo: 0.8s
```

#### **Ejemplo 2: Escalado ×4** 
```
Input:  128×128 → Output: 512×512  
Modelo: SwinIR
PSNR:   25.2 dB
SSIM:   0.82
Tiempo: 1.2s
```

## 🔍 **Características Técnicas**

### **Modelos Integrados**
- **ESRGAN**: Mejor balance velocidad/calidad
- **SwinIR**: Máxima calidad, mayor tiempo procesamiento  
- **EDSR**: Baseline estable y confiable

### **Formatos Soportados**
- **Entrada**: PNG, JPEG, TIFF
- **Salida**: PNG, JPEG (configurable)
- **Resoluciones**: 64×64 hasta 2048×2048

### **Métricas Calculadas**
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Tiempo de procesamiento**: Latencia en segundos
- **Factor de mejora**: Comparación visual

## 🚀 **Despliegue y Producción**

### **Configuración para Producción**

#### **Backend (FastAPI)**
```bash
# Usar gunicorn para producción
pip install gunicorn

# Ejecutar con múltiples workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### **Frontend (Streamlit)**
```bash
# Configurar para acceso externo
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### **Variables de Entorno**
```bash
# .env file
BACKEND_URL=http://localhost:8000
MODEL_PATH=../shared/models
MAX_IMAGE_SIZE=2048
ENABLE_GPU=true
DEBUG_MODE=false
```

## 🧪 **Testing y Validación**

### **Imágenes de Prueba**
La aplicación incluye un conjunto de imágenes de ejemplo:
```
shared/sample_images/
├── low_res_64x64.png
├── low_res_128x128.png  
├── low_res_256x256.png
└── histopatologia_sample.jpg
```

### **Validación de Modelos**
```python
# Test básico de funcionamiento
def test_model_loading():
    assert load_esrgan_model() is not None
    assert load_swinir_model() is not None
    assert load_edsr_model() is not None

def test_image_processing():
    test_image = load_test_image()
    result = apply_super_resolution(test_image, "esrgan", 2)
    assert result.size == (test_image.size[0]*2, test_image.size[1]*2)
```

## 🤝 **Contribución y Desarrollo**

### **Estructura de Desarrollo**
```bash
# Setup desarrollo completo
git clone https://github.com/JuanHoKKeR/App_SR_Microscopy.git
cd App_SR_Microscopy

# Backend setup
python -m venv backend_env
source backend_env/bin/activate
cd backend && pip install -r requirements.txt

# Frontend setup (nueva terminal)
python -m venv frontend_env  
source frontend_env/bin/activate
cd frontend && pip install -r requirements.txt
```

### **Agregar Nuevos Modelos**
1. Colocar archivo `.pth` en `shared/models/`
2. Actualizar función de carga en backend
3. Agregar opción en selector de frontend

## 📄 **Licencia**

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 **Reconocimientos**

- **FastAPI**: Por el excelente framework de API
- **Streamlit**: Por la simplificación de interfaces web en Python
- **PyTorch**: Por el framework de deep learning
- **Modelos Base**: ESRGAN, SwinIR, EDSR por sus arquitecturas

## 📞 **Contacto**

- **Autor**: Juan David Cruz Useche
- **Proyecto**: Trabajo de Grado - Super-Resolución para Histopatología  
- **GitHub**: [@JuanHoKKeR](https://github.com/JuanHoKKeR)
- **Repositorio**: [App_SR_Microscopy](https://github.com/JuanHoKKeR/App_SR_Microscopy)

---

## 🚨 **Nota de Desarrollo**

Esta aplicación se encuentra en **fase de desarrollo activo**. Algunas características pueden cambiar y se recomienda usar para propósitos de demostración y evaluación. Para uso en producción, contactar al desarrollador.

**⭐ Si encuentras útil esta demo, considera darle una estrella al repositorio!**
