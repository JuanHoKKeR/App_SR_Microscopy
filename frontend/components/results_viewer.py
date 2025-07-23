"""
Visualizador de resultados corregido
Muestra imágenes en tamaños proporcionales y métricas correctas
"""

import streamlit as st
import base64
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import io
import logging

from .ui_config import show_info_box, show_metric_card

logger = logging.getLogger(__name__)

class ResultsViewer:
    """Visualizador de resultados corregido"""
    
    def __init__(self):
        pass
    
    def display_scale_results(self, result: Dict[str, Any]):
        """Muestra resultados de procesamiento por escala"""
        if not result or not result.get("success", False):
            st.error("❌ No hay resultados válidos para mostrar")
            return
        
        st.markdown('<h2 class="sub-header">🎉 Resultados del Procesamiento</h2>', unsafe_allow_html=True)
        
        # Resumen del procesamiento
        self._show_processing_summary(result)
        
        # Comparación con tamaños proporcionales
        self._show_proportional_comparison(result)
        
        # Métricas de calidad
        self._show_quality_metrics_enhanced(result)
        
        # Progresión paso a paso
        self._show_step_progression(result)
        
        # Opciones de descarga
        self._show_download_options(result)
    
    def _show_processing_summary(self, result: Dict[str, Any]):
        """Muestra resumen del procesamiento realizado"""
        st.markdown("### 📊 Resumen del Procesamiento")
        
        # Obtener información de configuración
        config = result.get("config", {})
        selection = result.get("selection", {})
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_metric_card(
                config.get("architecture", result.get("architecture", "N/A")),
                "Arquitectura"
            )
        
        with col2:
            show_metric_card(
                f"×{config.get('target_scale', result.get('target_scale', 'N/A'))}",
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
        **📍 Región procesada:** ({selection.get('x', 0)}, {selection.get('y', 0)}) - {selection.get('width', 0)}×{selection.get('height', 0)} px<br>
        **🎯 Escalado conseguido:** {config.get('target_scale', result.get('target_scale', 'N/A'))}x
        """, "success")
    
    def _show_proportional_comparison(self, result: Dict[str, Any]):
        """Muestra comparación con tamaños proporcionales reales"""
        st.markdown("### 🔍 Comparación con Tamaños Proporcionales")
        
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
        
        # Obtener dimensiones reales
        config = result.get("config", {})
        target_scale = config.get("target_scale", result.get("target_scale", 2))
        
        # Layout de comparación con tamaños proporcionales
        col1, col2 = st.columns([1, target_scale])  # Columnas proporcionales al factor de escala
        
        with col1:
            st.markdown("#### 📷 Imagen Original")
            st.image(original_img, caption=f"Original: {original_img.size[0]}×{original_img.size[1]} px")
            
            # Información adicional
            st.markdown(f"""
            **Tamaño real:** {original_img.size[0]} × {original_img.size[1]} píxeles  
            **Área:** {original_img.size[0] * original_img.size[1]:,} píxeles²
            """)
        
        with col2:
            st.markdown(f"#### 🚀 Resultado (×{target_scale})")
            st.image(final_img, caption=f"Procesado: {final_img.size[0]}×{final_img.size[1]} px")
            
            # Información adicional
            gain_factor = (final_img.size[0] * final_img.size[1]) / (original_img.size[0] * original_img.size[1])
            st.markdown(f"""
            **Tamaño real:** {final_img.size[0]} × {final_img.size[1]} píxeles  
            **Área:** {final_img.size[0] * final_img.size[1]:,} píxeles²  
            **Ganancia:** {gain_factor:.1f}× más píxeles
            """)
        
        # Mostrar diferencia visual de tamaño
        st.markdown("#### 📏 Comparación Visual de Tamaño")
        st.markdown("""
        💡 **Nota:** Las columnas están dimensionadas proporcionalmente al factor de escala 
        para mostrar la diferencia real de tamaño entre las imágenes.
        """)
    
    def _show_quality_metrics_enhanced(self, result: Dict[str, Any]):
        """Muestra métricas de calidad mejoradas"""
        quality_metrics = result.get("quality_metrics")
        
        if not quality_metrics:
            st.markdown("### 📈 Métricas de Calidad")
            st.info("ℹ️ No se calcularon métricas de calidad para este procesamiento")
            return
        
        if "error" in quality_metrics:
            show_info_box(f"""
            ⚠️ **Error en evaluación de calidad:** {quality_metrics['error']}<br>
            Las métricas de calidad no están disponibles para este resultado.
            """, "warning")
            return
        
        st.markdown("### 📈 Métricas de Calidad")
        
        # Métricas principales
        col1, col2, col3 = st.columns(3)
        
        metrics = quality_metrics.get("metrics") or quality_metrics
        interpretation = quality_metrics.get("interpretation", {})
        
        with col1:
            psnr_val = metrics.get("psnr", -1)
            if psnr_val > 0:
                # Determinar color basado en calidad
                if psnr_val > 30:
                    psnr_color = "🟢"
                elif psnr_val > 25:
                    psnr_color = "🟡"
                else:
                    psnr_color = "🔴"
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{psnr_color} {psnr_val:.2f} dB</div>
                    <div class="metric-label">PSNR - {interpretation.get('psnr', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            ssim_val = metrics.get("ssim", -1)
            if ssim_val > 0:
                # Determinar color basado en calidad
                if ssim_val > 0.9:
                    ssim_color = "🟢"
                elif ssim_val > 0.7:
                    ssim_color = "🟡"
                else:
                    ssim_color = "🔴"
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{ssim_color} {ssim_val:.4f}</div>
                    <div class="metric-label">SSIM - {interpretation.get('ssim', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            perceptual_val = metrics.get("perceptual_index", -1)
            if perceptual_val >= 0:
                # Determinar color basado en calidad (menor es mejor para perceptual)
                if perceptual_val < 0.001:
                    perc_color = "🟢"
                elif perceptual_val < 0.01:
                    perc_color = "🟡"
                else:
                    perc_color = "🔴"
                
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{perc_color} {perceptual_val:.6f}</div>
                    <div class="metric-label">Índice Perceptual - {interpretation.get('perceptual', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Información sobre KimiaNet
        kimianet_used = quality_metrics.get("kimianet_used", False)
        if kimianet_used:
            show_info_box("""
            🧠 **KimiaNet Utilizado:** Las métricas perceptuales se calcularon usando DenseNet121 
            con pesos KimiaNet, específicamente entrenado para imágenes de histopatología de cáncer de mama.
            Esto proporciona una evaluación más relevante para tu dominio específico.
            """, "success")
        else:
            show_info_box("""
            ⚠️ **KimiaNet No Disponible:** Las métricas PSNR y SSIM están disponibles, 
            pero el índice perceptual especializado no se pudo calcular.
            """, "warning")
        
        # Interpretación detallada
        with st.expander("📚 Interpretación de Métricas"):
            st.markdown("""
            **PSNR (Peak Signal-to-Noise Ratio):**
            - 🟢 > 30 dB: Excelente calidad, diferencias imperceptibles
            - 🟡 25-30 dB: Buena calidad, diferencias mínimas
            - 🔴 < 25 dB: Calidad aceptable pero con artefactos visibles
            
            **SSIM (Structural Similarity Index):**
            - 🟢 > 0.9: Excelente preservación de estructura
            - 🟡 0.7-0.9: Buena preservación de estructura  
            - 🔴 < 0.7: Pérdida notable de estructura
            
            **Índice Perceptual (KimiaNet):**
            - 🟢 < 0.001: Excelente similitud perceptual
            - 🟡 0.001-0.01: Buena similitud perceptual
            - 🔴 > 0.01: Diferencias perceptuales notables
            
            💡 **Para histopatología de cáncer de mama**, el índice perceptual KimiaNet 
            es especialmente relevante ya que evalúa características específicas del dominio.
            """)
    
    def _show_step_progression(self, result: Dict[str, Any]):
        """Muestra la progresión paso a paso del procesamiento"""
        st.markdown("### 🔄 Progresión Paso a Paso")
        
        steps = result.get("steps", [])
        if not steps:
            st.warning("No hay información de pasos disponible")
            return
        
        # Mostrar pasos en columnas si son pocos, tabs si son muchos
        if len(steps) <= 3:
            cols = st.columns(len(steps))
            for i, step in enumerate(steps):
                with cols[i]:
                    self._show_single_step(step, i + 1)
        else:
            # Usar tabs para muchos pasos
            tab_names = [f"Paso {step['step']}" for step in steps]
            tabs = st.tabs(tab_names)
            
            for tab, step in zip(tabs, steps):
                with tab:
                    self._show_single_step(step, step['step'], detailed=True)
    
    def _show_single_step(self, step: Dict[str, Any], step_number: int, detailed: bool = False):
        """Muestra un paso individual del procesamiento"""
        st.markdown(f"**Paso {step_number}: {step.get('model_name', 'Modelo desconocido')}**")
        
        # Mostrar imagen del paso
        if "enhanced_patch" in step:
            image_data = self._base64_to_image(step["enhanced_patch"])
            if image_data is not None:
                st.image(image_data, 
                        caption=f"Salida: {step.get('output_size', 'N/A')}",
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
    
    def _show_download_options(self, result: Dict[str, Any]):
        """Muestra opciones de descarga"""
        st.markdown("### 💾 Opciones de Descarga")
        
        col1, col2, col3 = st.columns(3)
        
        config = result.get("config", {})
        
        with col1:
            # Descargar resultado final
            if "final_result" in result:
                final_bytes = self._base64_to_bytes(result["final_result"])
                if final_bytes:
                    st.download_button(
                        label="📥 Descargar Resultado Final",
                        data=final_bytes,
                        file_name=f"enhanced_x{config.get('target_scale', 'N')}_{config.get('architecture', 'unknown').lower()}.png",
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
            # Generar reporte de métricas
            if result.get("quality_metrics"):
                report_text = self._generate_metrics_report(result)
                st.download_button(
                    label="📄 Descargar Reporte",
                    data=report_text,
                    file_name="quality_report.txt",
                    mime="text/plain"
                )
    
    def _generate_metrics_report(self, result: Dict[str, Any]) -> str:
        """Genera reporte de texto con las métricas"""
        config = result.get("config", {})
        metrics = result.get("quality_metrics", {}).get("metrics", {})
        
        report_lines = [
            "=== REPORTE DE SUPER-RESOLUCIÓN ===",
            f"Fecha: {self._get_current_timestamp()}",
            "",
            "CONFIGURACIÓN:",
            f"- Arquitectura: {config.get('architecture', 'N/A')}",
            f"- Factor de escala: ×{config.get('target_scale', 'N/A')}",
            f"- Tamaño de parche: {config.get('patch_size', 'N/A')}×{config.get('patch_size', 'N/A')} px",
            f"- Modelos utilizados: {len(result.get('upsampling_path', []))} pasos",
            "",
            "MÉTRICAS DE CALIDAD:",
            f"- PSNR: {metrics.get('psnr', 'N/A'):.4f} dB",
            f"- SSIM: {metrics.get('ssim', 'N/A'):.6f}",
            f"- Índice Perceptual: {metrics.get('perceptual_index', 'N/A'):.8f}",
            "",
            "PROCESAMIENTO:",
            f"- Ruta: {' → '.join(result.get('upsampling_path', []))}",
            f"- Resolución inicial: {result.get('original_size', 'N/A')}",
            f"- Resolución final: {result.get('final_size', 'N/A')}",
        ]
        
        return "\n".join(report_lines)
    
    def display_evaluation_results(self, result: Dict[str, Any], show_difference_map: bool = False):
        """Muestra resultados de evaluación independiente"""
        st.markdown("### 📊 Resultados de Evaluación")
        
        if not result.get("success", False):
            st.error("❌ Error en la evaluación")
            return
        
        # Mostrar métricas principales
        metrics = result.get("metrics", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            psnr = metrics.get("psnr", 0)
            show_metric_card(f"{psnr:.2f} dB", "PSNR")
        
        with col2:
            ssim = metrics.get("ssim", 0)
            show_metric_card(f"{ssim:.4f}", "SSIM")
        
        with col3:
            perceptual = metrics.get("perceptual_index", -1)
            if perceptual >= 0:
                show_metric_card(f"{perceptual:.6f}", "Índice Perceptual")
            else:
                show_metric_card("N/A", "Índice Perceptual")
        
        # Interpretación
        interpretation = result.get("interpretation", {})
        if interpretation:
            st.markdown("**Interpretación:**")
            for metric, interp in interpretation.items():
                st.markdown(f"- **{metric.upper()}:** {interp}")
    
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
    
    def _get_current_timestamp(self) -> str:
        """Obtiene timestamp actual"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")