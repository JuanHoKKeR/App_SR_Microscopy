"""
Componente de procesamiento de imágenes para la UI
Maneja la lógica de llamadas a la API y progreso de procesamiento
"""

import streamlit as st
import time
import base64
import io
from typing import Optional, Dict, Any
import logging

from .api_client import APIClient
from .ui_config import show_progress_steps, show_info_box

logger = logging.getLogger(__name__)

class ImageProcessorUI:
    """Interfaz de usuario para procesamiento de imágenes"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def process_sequential_upsampling(self, 
                                    uploaded_file, 
                                    selection: Dict[str, int], 
                                    config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Procesa upsampling secuencial con UI de progreso"""
        
        architecture = config["architecture"]
        patch_size = config["patch_size"]
        target_scale = config["target_scale"]
        path_info = config["path_info"]
        
        # Preparar pasos para mostrar progreso
        steps = [f"Aplicar {model}" for model in path_info["path"]]
        steps.insert(0, "Preparando imagen")
        steps.append("Finalizando")
        
        # Contenedor de progreso
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            st.markdown('<h3 class="sub-header">🔄 Procesamiento en Curso</h3>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
        # Mostrar pasos
        with status_container:
            show_progress_steps(steps, 0)
            steps_display = st.empty()
        
        try:
            # Paso 1: Preparando imagen
            status_text.text("📁 Preparando imagen...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            # Actualizar pasos
            with steps_display:
                show_progress_steps(steps, 1)
            
            # Llamar a la API
            status_text.text("🚀 Enviando a procesamiento...")
            progress_bar.progress(20)
            
            # Resetear archivo
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            
            # Procesar con la API
            result = self.api_client.process_sequential_upsampling(
                image_file=uploaded_file,
                architecture=architecture,
                start_size=patch_size,
                target_scale=target_scale,
                x=selection["x"],
                y=selection["y"], 
                width=selection["width"],
                height=selection["height"]
            )
            
            if not result:
                st.error("❌ Error en el procesamiento")
                return None
            
            # Simular progreso por pasos
            total_steps = len(path_info["path"])
            for i in range(total_steps):
                progress = 20 + (70 * (i + 1) / total_steps)
                progress_bar.progress(int(progress))
                status_text.text(f"🔧 Procesando con {path_info['path'][i]}...")
                
                # Actualizar visualización de pasos
                with steps_display:
                    show_progress_steps(steps, i + 2)  # +2 porque empezamos en paso 1
                
                time.sleep(0.3)  # Pequeña pausa para visualización
            
            # Finalizar
            progress_bar.progress(95)
            status_text.text("✨ Finalizando...")
            
            with steps_display:
                show_progress_steps(steps, len(steps))
            
            time.sleep(0.5)
            progress_bar.progress(100)
            status_text.text("✅ ¡Procesamiento completado!")
            
            # Limpiar después de un momento
            time.sleep(1)
            progress_container.empty()
            status_container.empty()
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error durante el procesamiento: {e}")
            logger.error(f"Error en procesamiento: {e}")
            return None
    
    def process_single_patch(self, 
                           uploaded_file, 
                           selection: Dict[str, int], 
                           model_name: str) -> Optional[Dict[str, Any]]:
        """Procesa un parche individual"""
        
        # UI de progreso simple
        with st.spinner(f"Procesando con {model_name}..."):
            # Resetear archivo
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            
            result = self.api_client.process_patch(
                image_file=uploaded_file,
                model_name=model_name,
                x=selection["x"],
                y=selection["y"],
                width=selection["width"],
                height=selection["height"]
            )
            
            if result:
                st.success(f"✅ Procesamiento completado con {model_name}")
                return result
            else:
                st.error(f"❌ Error procesando con {model_name}")
                return None
    
    def show_processing_options(self, config: Dict[str, Any]) -> str:
        """Muestra opciones de procesamiento disponibles"""
        st.markdown("**🎛️ Opciones de Procesamiento:**")
        
        # Opción por defecto: secuencial
        processing_mode = st.radio(
            "Modo de procesamiento:",
            [
                "🔄 Secuencial (Recomendado)",
                "⚡ Modelo individual"
            ],
            help="Secuencial aplica múltiples modelos para alcanzar el factor de escala deseado"
        )
        
        if "individual" in processing_mode.lower():
            # Selección de modelo individual
            path_info = config["path_info"]
            available_models = path_info["path"]
            
            selected_model = st.selectbox(
                "Modelo a usar:",
                available_models,
                help="Selecciona un modelo específico (escalará x2)"
            )
            
            return f"individual:{selected_model}"
        
        return "sequential"
    
    def estimate_processing_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula información estimada del procesamiento"""
        path_info = config["path_info"]
        num_steps = len(path_info["path"])
        
        # Estimaciones basadas en experiencia
        estimated_time = num_steps * 2.5  # segundos por paso
        memory_usage = config["patch_size"] * config["patch_size"] * 3 * 4 / (1024**2)  # MB aprox
        
        return {
            "steps": num_steps,
            "estimated_time": estimated_time,
            "memory_usage": memory_usage,
            "input_resolution": f"{config['patch_size']}x{config['patch_size']}",
            "output_resolution": f"{config['patch_size'] * config['target_scale']}x{config['patch_size'] * config['target_scale']}"
        }
    
    def show_processing_preview(self, config: Dict[str, Any], selection: Dict[str, int]):
        """Muestra vista previa del procesamiento a realizar"""
        st.markdown('<h3 class="sub-header">📋 Resumen del Procesamiento</h3>', unsafe_allow_html=True)
        
        processing_info = self.estimate_processing_info(config)
        
        # Crear columnas para información
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Configuración:**")
            st.markdown(f"- **Arquitectura:** {config['architecture']}")
            st.markdown(f"- **Región:** ({selection['x']}, {selection['y']}) - {selection['width']}×{selection['height']}px")
            st.markdown(f"- **Tamaño de parche:** {config['patch_size']}×{config['patch_size']}px")
            st.markdown(f"- **Factor de escala:** ×{config['target_scale']}")
        
        with col2:
            st.markdown("**📊 Estimaciones:**")
            st.markdown(f"- **Pasos requeridos:** {processing_info['steps']}")
            st.markdown(f"- **Tiempo estimado:** ~{processing_info['estimated_time']:.1f}s")
            st.markdown(f"- **Resolución final:** {processing_info['output_resolution']}px")
            st.markdown(f"- **Uso de memoria:** ~{processing_info['memory_usage']:.1f}MB")
        
        # Mostrar ruta de procesamiento
        with st.expander("🛤️ Ver ruta de procesamiento detallada"):
            path_info = config["path_info"]
            current_size = config["patch_size"]
            
            for i, model_name in enumerate(path_info["path"]):
                next_size = current_size * 2
                st.markdown(f"**Paso {i+1}:** {model_name}")
                st.markdown(f"   📥 Entrada: {current_size}×{current_size}px → 📤 Salida: {next_size}×{next_size}px")
                current_size = next_size
        
        return processing_info
    
    def validate_processing_requirements(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida que se cumplan los requisitos para el procesamiento"""
        try:
            # Verificar que todos los modelos estén disponibles
            path_info = config["path_info"]
            
            if not path_info.get("all_models_available", False):
                return False, "No todos los modelos requeridos están disponibles"
            
            # Verificar configuración válida
            if config["patch_size"] <= 0:
                return False, "Tamaño de parche inválido"
            
            if config["target_scale"] < 2:
                return False, "Factor de escala debe ser al menos x2"
            
            # Verificar recursos del sistema (opcional)
            stats = self.api_client.get_stats()
            if stats and "memory_usage" in stats:
                memory_info = stats["memory_usage"]
                # Verificar memoria GPU si está disponible
                if "gpu_memory_free" in memory_info and memory_info["gpu_memory_free"] < 0.5:
                    return False, "Memoria GPU insuficiente (menos de 500MB libres)"
            
            return True, "Requisitos cumplidos"
            
        except Exception as e:
            logger.error(f"Error validando requisitos: {e}")
            return False, f"Error de validación: {e}"
    
    def show_processing_warnings(self, config: Dict[str, Any]):
        """Muestra advertencias relevantes para el procesamiento"""
        warnings = []
        
        # Advertencias por tamaño
        final_size = config["patch_size"] * config["target_scale"]
        if final_size > 2048:
            warnings.append(f"⚠️ Imagen final será muy grande ({final_size}×{final_size}px)")
        
        # Advertencias por número de pasos
        num_steps = len(config["path_info"]["path"])
        if num_steps > 4:
            warnings.append(f"⚠️ Procesamiento largo ({num_steps} pasos)")
        
        # Advertencias por arquitectura
        if config["architecture"].upper() == "ESRGAN":
            warnings.append("💡 ESRGAN puede generar detalles muy finos pero tomar más tiempo")
        elif config["architecture"].upper() == "SWINIR":
            warnings.append("💡 SwinIR ofrece buen balance entre calidad y velocidad")
        
        # Mostrar advertencias
        if warnings:
            with st.expander("⚠️ Consideraciones importantes"):
                for warning in warnings:
                    st.markdown(f"- {warning}")
    
    def base64_to_image_bytes(self, base64_str: str) -> bytes:
        """Convierte string base64 a bytes de imagen"""
        try:
            img_data = base64.b64decode(base64_str)
            return img_data
        except Exception as e:
            logger.error(f"Error convirtiendo base64: {e}")
            return b""