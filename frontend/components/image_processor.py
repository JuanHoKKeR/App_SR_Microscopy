"""
Componente corregido para procesamiento de imágenes por factor de escala
"""

import streamlit as st
import time
import base64
import io
from typing import Optional, Dict, Any, Tuple
import logging

from .api_client import APIClient
from .ui_config import show_progress_steps, show_info_box

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Procesador de imágenes corregido para trabajar con factores de escala"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def process_by_scale(self, 
                        uploaded_file, 
                        selection: Dict[str, int], 
                        config: Dict[str, Any],
                        evaluate_quality: bool = False) -> Optional[Dict[str, Any]]:
        """Procesa por factor de escala usando cascada de modelos"""
        
        architecture = config["architecture"]
        patch_size = config["patch_size"]
        target_scale = config["target_scale"]
        processing_path = config["processing_path"]
        
        # Preparar pasos para mostrar progreso
        steps = ["Preparando imagen"]
        for model in processing_path:
            steps.append(f"Aplicando {model}")
        steps.append("Finalizando")
        
        # Contenedor de progreso
        progress_container = st.container()
        
        with progress_container:
            st.markdown('<h4>🔄 Procesamiento en Curso</h4>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Mostrar pasos
            steps_display = st.empty()
            with steps_display:
                show_progress_steps(steps, 0)
        
        try:
            # Paso 1: Preparando imagen
            status_text.text("📁 Preparando imagen...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            # Actualizar pasos
            with steps_display:
                show_progress_steps(steps, 1)
            
            # Llamar a la API para procesamiento secuencial
            status_text.text("🚀 Enviando a procesamiento...")
            progress_bar.progress(20)
            
            # Resetear archivo
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            
            # Procesar con la API usando el endpoint de procesamiento secuencial
            result = self.api_client.process_sequential_upsampling(
                image_file=uploaded_file,
                architecture=architecture,
                start_size=patch_size,
                target_scale=target_scale,
                x=selection["x"],
                y=selection["y"], 
                width=selection["width"],
                height=selection["height"],
                evaluate_quality=evaluate_quality
            )
            
            if not result:
                st.error("❌ Error en el procesamiento")
                return None
            
            # Simular progreso por pasos
            total_steps = len(processing_path)
            for i in range(total_steps):
                progress = 20 + (70 * (i + 1) / total_steps)
                progress_bar.progress(int(progress))
                status_text.text(f"🔧 Procesando con {processing_path[i]}...")
                
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
            
            # Añadir información adicional al resultado
            result["config"] = config
            result["selection"] = selection
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error durante el procesamiento: {e}")
            logger.error(f"Error en procesamiento: {e}")
            return None
    
    def validate_processing_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Valida la configuración de procesamiento"""
        try:
            # Verificar que todos los modelos estén disponibles
            processing_path = config["processing_path"]
            
            if not processing_path:
                return False, "No se encontró una ruta de procesamiento válida"
            
            # Verificar cada modelo en la ruta
            for model_name in processing_path:
                if not self.api_client.is_model_loaded(model_name):
                    return False, f"Modelo {model_name} no está disponible"
            
            # Verificar configuración válida
            if config["patch_size"] <= 0:
                return False, "Tamaño de parche inválido"
            
            if config["target_scale"] < 2:
                return False, "Factor de escala debe ser al menos x2"
            
            return True, "Configuración válida"
            
        except Exception as e:
            logger.error(f"Error validando configuración: {e}")
            return False, f"Error de validación: {e}"
    
    def estimate_processing_time(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estima información del procesamiento"""
        processing_path = config["processing_path"]
        num_steps = len(processing_path)
        
        # Estimaciones basadas en experiencia
        estimated_time = num_steps * 2.5  # segundos por paso
        
        return {
            "steps": num_steps,
            "estimated_time": estimated_time,
            "input_resolution": f"{config['patch_size']}x{config['patch_size']}",
            "output_resolution": f"{config['final_size']}x{config['final_size']}",
            "scale_factor": config["target_scale"],
            "architecture": config["architecture"]
        }
    
    def show_processing_preview(self, config: Dict[str, Any], selection: Dict[str, int]):
        """Muestra vista previa del procesamiento a realizar"""
        st.markdown('<h4>📋 Resumen del Procesamiento</h4>', unsafe_allow_html=True)
        
        processing_info = self.estimate_processing_time(config)
        
        # Crear columnas para información
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Configuración:**")
            st.markdown(f"- **Arquitectura:** {config['architecture']}")
            st.markdown(f"- **Región:** ({selection['x']}, {selection['y']}) - {selection['width']}×{selection['height']}px")
            st.markdown(f"- **Factor de escala:** ×{config['target_scale']}")
        
        with col2:
            st.markdown("**📊 Estimaciones:**")
            st.markdown(f"- **Pasos requeridos:** {processing_info['steps']}")
            st.markdown(f"- **Tiempo estimado:** ~{processing_info['estimated_time']:.1f}s")
            st.markdown(f"- **Resolución final:** {processing_info['output_resolution']}px")
        
        # Mostrar ruta de procesamiento
        with st.expander("🛤️ Ver ruta de procesamiento detallada"):
            processing_path = config["processing_path"]
            current_size = config["patch_size"]
            
            for i, model_name in enumerate(processing_path):
                next_size = current_size * 2
                st.markdown(f"**Paso {i+1}:** {model_name}")
                st.markdown(f"   📥 Entrada: {current_size}×{current_size}px → 📤 Salida: {next_size}×{next_size}px")
                current_size = next_size
        
        return processing_info
    
    def show_processing_warnings(self, config: Dict[str, Any]):
        """Muestra advertencias relevantes para el procesamiento"""
        warnings = []
        
        # Advertencias por tamaño final
        final_size = config["final_size"]
        if final_size > 2048:
            warnings.append(f"⚠️ Imagen final será muy grande ({final_size}×{final_size}px)")
        
        # Advertencias por número de pasos
        num_steps = len(config["processing_path"])
        if num_steps > 4:
            warnings.append(f"⚠️ Procesamiento largo ({num_steps} pasos)")
        
        # Advertencias por arquitectura
        if config["architecture"].upper() == "ESRGAN":
            warnings.append("💡 ESRGAN genera detalles muy finos pero puede tomar más tiempo")
        elif config["architecture"].upper() == "SWINIR":
            warnings.append("💡 SwinIR ofrece buen balance entre calidad y velocidad")
        elif config["architecture"].upper() == "EDSR":
            warnings.append("💡 EDSR es eficiente y produce resultados suaves")
        
        # Mostrar advertencias
        if warnings:
            with st.expander("⚠️ Consideraciones importantes"):
                for warning in warnings:
                    st.markdown(f"- {warning}")
    
    def get_scale_capabilities(self, architecture: str, available_models: list) -> Dict[str, Any]:
        """Obtiene las capacidades de escala para una arquitectura"""
        arch_models = [m for m in available_models 
                      if m["architecture"].upper() == architecture.upper() and m["available"]]
        
        if not arch_models:
            return {"available_sizes": [], "max_scales": {}}
        
        # Obtener tamaños de entrada disponibles
        input_sizes = sorted(list(set([m["input_size"] for m in arch_models])))
        
        # Calcular escalas máximas para cada tamaño
        max_scales = {}
        for size in input_sizes:
            max_scale = self._calculate_max_scale_for_size(size, arch_models)
            max_scales[size] = max_scale
        
        return {
            "available_sizes": input_sizes,
            "max_scales": max_scales,
            "total_models": len(arch_models)
        }
    
    def _calculate_max_scale_for_size(self, start_size: int, arch_models: list) -> int:
        """Calcula la escala máxima posible desde un tamaño dado"""
        max_scale = 1
        current_size = start_size
        
        while True:
            next_size = current_size * 2
            # Buscar si existe un modelo que vaya de current_size a next_size
            model_exists = any(
                m["input_size"] == current_size and m["output_size"] == next_size 
                for m in arch_models
            )
            
            if model_exists:
                max_scale *= 2
                current_size = next_size
            else:
                break
        
        return max_scale