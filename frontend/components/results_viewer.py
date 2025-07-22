"""
Componente para visualización de resultados
Maneja la presentación de imágenes procesadas y comparaciones
"""

import streamlit as st
import base64
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import io
import logging

from .ui_config import show_comparison_layout, show_info_box, show_metric_card

logger = logging.getLogger(__name__)

class ResultsViewer:
    """Visualizador de resultados de superresolución"""
    
    def __init__(self):
        pass
    
    def display_sequential_results(self, result: Dict[str, Any]):
        """Muestra resultados de procesamiento secuencial"""
        if not result or not result.get("success", False):
            st.error("❌ No hay resultados válidos para mostrar")
            return
        
        st.markdown('<h2 class="sub-header">🎉 Resultados del Procesamiento</h2>', unsafe_allow_html=True)
        
        # Resumen de procesamiento
        self._show_processing_summary(result)
        
        # Progresión paso a paso
        self._show_step_progression(result)
        
        # Comparación final
        self._show_final_comparison(result)
        
        # Opciones de descarga
        self._show_download_options(result)
    
    def _show_processing_summary(self, result: Dict[str, Any]):
        """Muestra resumen del procesamiento realizado"""
        st.markdown("### 📊 Resumen del Procesamiento")
        
        # Métricas principales
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
        
        # Información detallada
        show_info_box(f"""
        **🛤️ Ruta de procesamiento:** {' → '.join(result.get('upsampling_path', []))}<br>
        **⏱️ Arquitectura utilizada:** {result.get('architecture', 'Desconocida')}<br>
        **🎯 Escalado conseguido:** {result.get('target_scale', 'N/A')}x
        """, "success")
    
    def _show_step_progression(self, result: Dict[str, Any]):
        """Muestra la progresión paso a paso del procesamiento"""
        st.markdown("### 🔄 Progresión Paso a Paso")
        
        steps = result.get("steps", [])
        if not steps:
            st.warning("No hay información de pasos disponible")
            return
        
        # Mostrar pasos en tabs para mejor organización
        if len(steps) <= 4:
            # Si hay pocos pasos, usar columnas
            cols = st.columns(len(steps))
            for i, step in enumerate(steps):
                with cols[i]:
                    self._show_single_step(step, i + 1)
        else:
            # Si hay muchos pasos, usar tabs
            tab_names = [f"Paso {step['step']}" for step in steps]
            tabs = st.tabs(tab_names)
            
            for i, (tab, step) in enumerate(zip(tabs, steps)):
                with tab:
                    self._show_single_step(step, i + 1, detailed=True)
    
    def _show_single_step(self, step: Dict[str, Any], step_number: int, detailed: bool = False):
        """Muestra un paso individual del procesamiento"""
        st.markdown(f"**Paso {step_number}**")
        st.markdown(f"*{step.get('model_name', 'Modelo desconocido')}*")
        
        # Mostrar imagen del paso
        if "enhanced_patch" in step:
            image_data = self._base64_to_image(step["enhanced_patch"])
            if image_data is not None:
                st.image(image_data, 
                        caption=f"{step.get('output_size', 'N/A')}",
                        use_column_width=True)
        
        if detailed:
            # Información adicional en modo detallado
            st.markdown(f"- **Entrada:** {step.get('input_size', 'N/A')}")
            st.markdown(f"- **Salida:** {step.get('output_size', 'N/A')}")
            
            # Información del modelo si está disponible
            model_config = step.get("model_config", {})
            if model_config:
                st.markdown(f"- **Tipo:** {model_config.get('type', 'N/A')}")
                if 'checkpoint_iter' in model_config:
                    st.markdown(f"- **Checkpoint:** {model_config['checkpoint_iter']}")
    
    def _show_final_comparison(self, result: Dict[str, Any]):
        """Muestra comparación final entre original y resultado"""
        st.markdown("### 🔍 Comparación Final")
        
        original_b64 = result.get("original_patch")
        final_b64 = result.get("final_result")
        
        if not original_b64 or not final_b64:
            st.error("❌ Imágenes de comparación no disponibles")
            return
        
        # Convertir de base64 a imágenes
        original_img = self._base64_to_image(original_b64)
        final_img = self._base64_to_image(final_b64)
        
        if original_img is None or final_img is None:
            st.error("❌ Error cargando imágenes para comparación")
            return
        
        # Layout de comparación
        show_comparison_layout(
            "Imagen Original",
            f"Resultado (x{result.get('target_scale', 'N/A')})",
            original_img,
            final_img
        )
        
        # Información de calidad (si está disponible)
        self._show_quality_metrics(result)
    
    def _show_quality_metrics(self, result: Dict[str, Any]):
        """Muestra métricas de calidad si están disponibles"""
        # Placeholder para métricas futuras
        with st.expander("📈 Métricas de Calidad (Próximamente)"):
            st.info("""
            🚧 **Funcionalidades futuras:**
            - PSNR (Peak Signal-to-Noise Ratio)
            - SSIM (Structural Similarity Index)
            - Índice Perceptual con KimiaNet
            - Detección de artefactos
            - Análisis de bordes y texturas
            """)
    
    def _show_download_options(self, result: Dict[str, Any]):
        """Muestra opciones de descarga"""
        st.markdown("### 💾 Opciones de Descarga")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Descargar resultado final
            if "final_result" in result:
                final_bytes = self._base64_to_bytes(result["final_result"])
                if final_bytes:
                    st.download_button(
                        label="📥 Descargar Resultado Final",
                        data=final_bytes,
                        file_name=f"enhanced_x{result.get('target_scale', 'N')}_{result.get('architecture', 'unknown').lower()}.png",
                        mime="image/png",
                        type="primary"
                    )
        
        with col2:
            # Descargar imagen original del parche
            if "original_patch" in result:
                original_bytes = self._base64_to_bytes(result["original_patch"])
                if original_bytes:
                    st.download_button(
                        label="📥 Descargar Original",
                        data=original_bytes,
                        file_name="original_patch.png",
                        mime="image/png"
                    )
        
        with col3:
            # Descargar reporte (futuro)
            if st.button("📄 Generar Reporte", help="Próximamente disponible"):
                st.info("🚧 Función de reporte en desarrollo")
    
    def _base64_to_image(self, base64_str: str) -> Optional[Image.Image]:
        """Convierte base64 a imagen PIL"""
        try:
            img_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(img_data))
            return image
        except Exception as e:
            logger.error(f"Error convirtiendo base64 a imagen: {e}")
            return None
    
    def _base64_to_bytes(self, base64_str: str) -> Optional[bytes]:
        """Convierte base64 a bytes"""
        try:
            return base64.b64decode(base64_str)
        except Exception as e:
            logger.error(f"Error convirtiendo base64 a bytes: {e}")
            return None
    
    def show_processing_comparison(self, results: List[Dict[str, Any]]):
        """Compara resultados de diferentes arquitecturas"""
        st.markdown("### ⚖️ Comparación de Arquitecturas")
        
        if len(results) < 2:
            st.info("Necesitas al menos 2 resultados para comparar")
            return
        
        # Crear tabs para cada resultado
        tab_names = [f"{r.get('architecture', 'N/A')} (x{r.get('target_scale', 'N')})" for r in results]
        tabs = st.tabs(tab_names)
        
        for tab, result in zip(tabs, results):
            with tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    if "final_result" in result:
                        final_img = self._base64_to_image(result["final_result"])
                        if final_img:
                            st.image(final_img, 
                                   caption=f"Resultado {result.get('architecture', 'N/A')}",
                                   use_column_width=True)
                
                with col2:
                    # Información del procesamiento
                    st.markdown("**📊 Información:**")
                    st.markdown(f"- **Arquitectura:** {result.get('architecture', 'N/A')}")
                    st.markdown(f"- **Escala:** ×{result.get('target_scale', 'N/A')}")
                    st.markdown(f"- **Pasos:** {len(result.get('steps', []))}")
                    st.markdown(f"- **Resolución:** {result.get('final_size', 'N/A')}")
    
    def show_batch_results(self, batch_results: List[Dict[str, Any]]):
        """Muestra resultados de procesamiento en lotes"""
        st.markdown("### 📦 Resultados del Lote")
        
        # Resumen del lote
        successful = sum(1 for r in batch_results if r.get("success", False))
        total = len(batch_results)
        
        show_info_box(f"""
        **📊 Resumen del lote:**<br>
        - Procesados exitosamente: {successful}/{total}<br>
        - Tasa de éxito: {(successful/total)*100:.1f}%
        """, "success" if successful == total else "warning")
        
        # Mostrar cada resultado
        for i, result in enumerate(batch_results):
            with st.expander(f"Imagen {i+1} - {'✅ Exitoso' if result.get('success') else '❌ Error'}"):
                if result.get("success"):
                    self.display_sequential_results(result)
                else:
                    st.error(f"Error: {result.get('error', 'Error desconocido')}")
    
    def export_comparison_report(self, results: List[Dict[str, Any]]) -> str:
        """Genera reporte de comparación en formato texto"""
        report_lines = [
            "# Reporte de Comparación de Super-Resolución",
            f"Generado en: {self._get_current_timestamp()}",
            "",
            "## Resumen"
        ]
        
        for i, result in enumerate(results):
            report_lines.extend([
                f"### Resultado {i+1}",
                f"- Arquitectura: {result.get('architecture', 'N/A')}",
                f"- Factor de escala: ×{result.get('target_scale', 'N/A')}",
                f"- Pasos aplicados: {len(result.get('steps', []))}",
                f"- Resolución final: {result.get('final_size', 'N/A')}",
                f"- Ruta de procesamiento: {' → '.join(result.get('upsampling_path', []))}",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def _get_current_timestamp(self) -> str:
        """Obtiene timestamp actual"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")