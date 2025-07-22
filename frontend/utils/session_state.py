"""
Utilidades para manejo del estado de sesión en Streamlit
"""

import streamlit as st
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Valores por defecto del estado de sesión
DEFAULT_SESSION_STATE = {
    "mode": "🎯 Selección de Parches",
    "architecture": "ESRGAN",
    "uploaded_image": None,
    "processed_results": [],
    "current_selection": None,
    "processing_history": [],
    "ui_preferences": {
        "show_advanced_options": False,
        "auto_preview": True,
        "show_processing_steps": True
    },
    "api_status": {
        "connected": False,
        "last_check": None,
        "available_models": []
    }
}

def init_session_state():
    """Inicializa el estado de sesión con valores por defecto"""
    for key, default_value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            logger.debug(f"Inicializado session_state['{key}'] = {default_value}")

def get_session_state(key: str, default: Any = None) -> Any:
    """Obtiene un valor del estado de sesión"""
    return st.session_state.get(key, default)

def update_session_state(key: str, value: Any):
    """Actualiza un valor en el estado de sesión"""
    st.session_state[key] = value
    logger.debug(f"Actualizado session_state['{key}'] = {value}")

def clear_session_state():
    """Limpia todo el estado de sesión"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    logger.info("Estado de sesión limpiado y reinicializado")

def reset_processing_state():
    """Resetea solo el estado relacionado con procesamiento"""
    processing_keys = [
        "processed_results",
        "current_selection", 
        "processing_history"
    ]
    
    for key in processing_keys:
        if key in st.session_state:
            st.session_state[key] = DEFAULT_SESSION_STATE.get(key, None)
    
    logger.info("Estado de procesamiento reseteado")

class SessionStateManager:
    """Gestor avanzado del estado de sesión"""
    
    @staticmethod
    def save_processing_result(result: Dict[str, Any]):
        """Guarda un resultado de procesamiento en el historial"""
        if "processed_results" not in st.session_state:
            st.session_state.processed_results = []
        
        # Agregar timestamp
        import datetime
        result["timestamp"] = datetime.datetime.now().isoformat()
        
        st.session_state.processed_results.append(result)
        
        # Limitar historial a últimos 10 resultados
        if len(st.session_state.processed_results) > 10:
            st.session_state.processed_results = st.session_state.processed_results[-10:]
        
        logger.info(f"Resultado de procesamiento guardado. Total: {len(st.session_state.processed_results)}")
    
    @staticmethod
    def get_processing_history() -> List[Dict[str, Any]]:
        """Obtiene el historial de procesamientos"""
        return get_session_state("processed_results", [])
    
    @staticmethod
    def save_image_upload(uploaded_file):
        """Guarda información de la imagen subida"""
        if uploaded_file is not None:
            image_info = {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "type": uploaded_file.type,
                "upload_time": st.session_state.get("upload_time")
            }
            update_session_state("uploaded_image", image_info)
            logger.info(f"Imagen subida guardada: {uploaded_file.name}")
    
    @staticmethod
    def save_selection_coordinates(coordinates: Dict[str, int]):
        """Guarda las coordenadas de selección actual"""
        update_session_state("current_selection", coordinates)
        logger.debug(f"Coordenadas guardadas: {coordinates}")
    
    @staticmethod
    def get_current_selection() -> Optional[Dict[str, int]]:
        """Obtiene la selección actual"""
        return get_session_state("current_selection")
    
    @staticmethod
    def update_api_status(connected: bool, models: List[Dict] = None):
        """Actualiza el estado de conexión con la API"""
        import datetime
        
        api_status = {
            "connected": connected,
            "last_check": datetime.datetime.now().isoformat(),
            "available_models": models or []
        }
        
        update_session_state("api_status", api_status)
        logger.info(f"Estado API actualizado: {'conectado' if connected else 'desconectado'}")
    
    @staticmethod
    def get_api_status() -> Dict[str, Any]:
        """Obtiene el estado actual de la API"""
        return get_session_state("api_status", DEFAULT_SESSION_STATE["api_status"])
    
    @staticmethod
    def toggle_advanced_options():
        """Alterna la visualización de opciones avanzadas"""
        current = get_session_state("ui_preferences", {}).get("show_advanced_options", False)
        
        ui_prefs = get_session_state("ui_preferences", {})
        ui_prefs["show_advanced_options"] = not current
        update_session_state("ui_preferences", ui_prefs)
        
        return not current
    
    @staticmethod
    def get_ui_preference(key: str, default: Any = None) -> Any:
        """Obtiene una preferencia de UI"""
        ui_prefs = get_session_state("ui_preferences", {})
        return ui_prefs.get(key, default)
    
    @staticmethod
    def set_ui_preference(key: str, value: Any):
        """Establece una preferencia de UI"""
        ui_prefs = get_session_state("ui_preferences", {})
        ui_prefs[key] = value
        update_session_state("ui_preferences", ui_prefs)
        logger.debug(f"Preferencia UI actualizada: {key} = {value}")
    
    @staticmethod
    def add_to_processing_history(action: str, details: Dict[str, Any]):
        """Agrega una acción al historial de procesamiento"""
        if "processing_history" not in st.session_state:
            st.session_state.processing_history = []
        
        import datetime
        history_entry = {
            "action": action,
            "details": details,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        st.session_state.processing_history.append(history_entry)
        
        # Limitar historial
        if len(st.session_state.processing_history) > 50:
            st.session_state.processing_history = st.session_state.processing_history[-50:]
        
        logger.debug(f"Acción agregada al historial: {action}")
    
    @staticmethod
    def export_session_data() -> Dict[str, Any]:
        """Exporta datos de la sesión para respaldo"""
        export_data = {}
        
        # Solo exportar ciertos campos
        export_keys = [
            "processed_results",
            "processing_history", 
            "ui_preferences"
        ]
        
        for key in export_keys:
            if key in st.session_state:
                export_data[key] = st.session_state[key]
        
        import datetime
        export_data["export_timestamp"] = datetime.datetime.now().isoformat()
        export_data["session_id"] = st.session_state.get("session_id", "unknown")
        
        return export_data
    
    @staticmethod
    def import_session_data(import_data: Dict[str, Any]):
        """Importa datos de sesión desde un respaldo"""
        try:
            # Validar datos de importación
            if not isinstance(import_data, dict):
                raise ValueError("Datos de importación inválidos")
            
            # Importar solo campos seguros
            safe_keys = [
                "processed_results",
                "ui_preferences"
            ]
            
            for key in safe_keys:
                if key in import_data:
                    update_session_state(key, import_data[key])
            
            logger.info("Datos de sesión importados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error importando datos de sesión: {e}")
            return False
    
    @staticmethod
    def get_session_summary() -> Dict[str, Any]:
        """Obtiene un resumen del estado actual de la sesión"""
        return {
            "mode": get_session_state("mode"),
            "architecture": get_session_state("architecture"),
            "has_uploaded_image": get_session_state("uploaded_image") is not None,
            "has_selection": get_session_state("current_selection") is not None,
            "results_count": len(get_session_state("processed_results", [])),
            "history_count": len(get_session_state("processing_history", [])),
            "api_connected": get_session_state("api_status", {}).get("connected", False)
        }
    
    @staticmethod
    def debug_session_state():
        """Función de debug para inspeccionar el estado de sesión"""
        if st.session_state.get("debug_mode", False):
            st.sidebar.markdown("---")
            st.sidebar.markdown("**🐛 Debug - Session State:**")
            
            with st.sidebar.expander("Ver estado completo"):
                for key, value in st.session_state.items():
                    if not key.startswith("_"):  # Omitir claves internas
                        st.write(f"**{key}:** {type(value).__name__}")
                        if isinstance(value, (str, int, float, bool)):
                            st.write(f"  → {value}")
                        elif isinstance(value, (list, dict)):
                            st.write(f"  → {len(value)} elementos")

# Alias para facilitar uso
session_manager = SessionStateManager()