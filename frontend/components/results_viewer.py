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
        quality_metrics = result.get("quality_metrics")
        
        if quality_metrics and "error" not in quality_metrics:
            st.markdown("### 📈 Métricas de Calidad")
            
            # Métricas principales
            col1, col2, col3 = st.columns(3)
            
            metrics = quality_metrics.get("metrics") or quality_metrics
            interpretation = quality_metrics.get("interpretation", {})
            
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
            
            # Información adicional
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
            
            # Explicación de métricas
            with st.expander("ℹ️ Explicación de Métricas"):
                st.markdown("""
                **PSNR (Peak Signal-to-Noise Ratio):**
                - Mide la relación entre señal y ruido
                - Valores más altos = mejor calidad
                - Típico: 20-35 dB para super-resolución
                
                **SSIM (Structural Similarity Index):**
                - Mide similitud estructural percibida
                - Rango: 0-1, valores más altos = mejor
                - Considera luminancia, contraste y estructura
                
                **Índice Perceptual (KimiaNet):**
                - Distancia en espacio de características KimiaNet
                - Valores más bajos = mayor similitud perceptual
                - Específicamente entrenado para histopatología
                """)
        
        elif quality_metrics and "error" in quality_metrics:
            show_info_box(f"""
            ⚠️ **Error en evaluación de calidad:** {quality_metrics['error']}<br>
            Las métricas de calidad no están disponibles para este resultado.
            """, "warning")
        
        else:
            # Placeholder para métricas futuras si no hay métricas
            with st.expander("📈 Métricas de Calidad (No disponibles)"):
                st.info("""
                🚧 **Para obtener métricas de calidad:**
                - Habilita "Evaluar Calidad" en la configuración
                - Asegúrate de que KimiaNet esté disponible
                - Las métricas incluyen PSNR, SSIM e Índice Perceptual
                """)
                
                # Botón para verificar KimiaNet
                if st.button("🔍 Verificar Estado de KimiaNet"):
                    # Esto se puede implementar para hacer una llamada a la API
                    st.info("Verificando estado de KimiaNet...")
    
    def show_quality_comparison(self, results: List[Dict[str, Any]]):
        """Compara métricas de calidad entre diferentes resultados"""
        st.markdown("### ⚖️ Comparación de Calidad")
        
        # Filtrar resultados que tienen métricas de calidad
        results_with_metrics = [
            r for r in results 
            if r.get("quality_metrics") and "error" not in r.get("quality_metrics", {})
        ]
        
        if len(results_with_metrics) < 2:
            st.info("Se necesitan al menos 2 resultados con métricas de calidad para comparar")
            return
        
        # Crear tabla comparativa
        comparison_data = []
        for result in results_with_metrics:
            metrics = result.get("quality_metrics", {}).get("metrics") or result.get("quality_metrics", {})
            
            comparison_data.append({
                "Arquitectura": result.get("architecture", "N/A"),
                "Escala": f"x{result.get('target_scale', 'N/A')}",
                "PSNR (dB)": f"{metrics.get('psnr', -1):.2f}" if metrics.get('psnr', -1) > 0 else "N/A",
                "SSIM": f"{metrics.get('ssim', -1):.4f}" if metrics.get('ssim', -1) > 0 else "N/A",
                "Índice Perceptual": f"{metrics.get('perceptual_index', -1):.6f}" if metrics.get('perceptual_index', -1) >= 0 else "N/A"
            })
        
        # Mostrar tabla
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Análisis automático
        if len(comparison_data) > 1:
            st.markdown("**🏆 Análisis Automático:**")
            
            # Encontrar mejores valores
            best_psnr = max([float(d["PSNR (dB)"].replace(" dB", "")) for d in comparison_data if d["PSNR (dB)"] != "N/A"])
            best_ssim = max([float(d["SSIM"]) for d in comparison_data if d["SSIM"] != "N/A"])
            
            best_psnr_arch = next(d["Arquitectura"] for d in comparison_data if d["PSNR (dB)"] != "N/A" and float(d["PSNR (dB)"].replace(" dB", "")) == best_psnr)
            best_ssim_arch = next(d["Arquitectura"] for d in comparison_data if d["SSIM"] != "N/A" and float(d["SSIM"]) == best_ssim)
            
            st.markdown(f"- **Mejor PSNR:** {best_psnr_arch} ({best_psnr:.2f} dB)")
            st.markdown(f"- **Mejor SSIM:** {best_ssim_arch} ({best_ssim:.4f})")
            
            # Análisis perceptual si está disponible
            perceptual_values = [float(d["Índice Perceptual"]) for d in comparison_data if d["Índice Perceptual"] != "N/A"]
            if perceptual_values:
                best_perceptual = min(perceptual_values)  # Menor es mejor
                best_perceptual_arch = next(d["Arquitectura"] for d in comparison_data if d["Índice Perceptual"] != "N/A" and float(d["Índice Perceptual"]) == best_perceptual)
                st.markdown(f"- **Mejor Índice Perceptual:** {best_perceptual_arch} ({best_perceptual:.6f})")
    
    def show_kimianet_info(self):
        """Muestra información sobre KimiaNet"""
        st.markdown("### 🧠 Acerca de KimiaNet")
        
        show_info_box("""
        **KimiaNet** es una red neuronal convolucional pre-entrenada específicamente para 
        imágenes de histopatología. Utiliza la arquitectura DenseNet121 y ha sido entrenada 
        en un gran dataset de imágenes médicas para extraer características relevantes 
        para el análisis de tejidos.
        
        **En esta aplicación:**
        - Se usa para calcular un índice perceptual especializado
        - Evalúa la similitud entre imágenes en el espacio de características médicas
        - Proporciona una métrica más relevante que PSNR/SSIM para histopatología
        """, "info")
    
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