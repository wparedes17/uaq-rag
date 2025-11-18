import json
import os
from typing import Dict, Any, Optional
from ..utils.file_manager import FileManager


class Settings:
    DEFAULT_SETTINGS = {
        "embedding_model": "text-embedding-3-small",
        "default_topic_documents": 5,
        "topic_mixing_threshold": 0.85,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "collection_name": "topics"
    }
    
    SETTINGS_FILE = "settings.json"
    
    def __init__(self):
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file or create defaults"""
        if os.path.exists(self.SETTINGS_FILE):
            try:
                return FileManager.load_json(self.SETTINGS_FILE)
            except Exception:
                pass
        
        # Create default settings file
        FileManager.save_json(self.DEFAULT_SETTINGS, self.SETTINGS_FILE)
        return self.DEFAULT_SETTINGS.copy()
    
    def save_settings(self) -> None:
        """Save current settings to file"""
        FileManager.save_json(self.settings, self.SETTINGS_FILE)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set setting value"""
        self.settings[key] = value
    
    def update_embedding_model(self, model: str) -> None:
        """Update the embedding model setting"""
        self.set("embedding_model", model)
        self.save_settings()
        print(f"✅ Embedding model updated to: {model}")
    
    def update_topic_documents(self, count: int) -> None:
        """Update default topic documents setting"""
        self.set("default_topic_documents", count)
        self.save_settings()
        print(f"✅ Default topic documents updated to: {count}")
    
    def update_mixing_threshold(self, threshold: float) -> None:
        """Update topic mixing threshold setting"""
        self.set("topic_mixing_threshold", threshold)
        self.save_settings()
        print(f"✅ Topic mixing threshold updated to: {threshold}")
    
    def display_settings(self) -> None:
        """Display current settings"""
        print("\n⚙️  Current Settings:")
        print(f"   🤖 Embedding Model: {self.get('embedding_model')}")
        print(f"   📄 Default Topic Documents: {self.get('default_topic_documents')}")
        print(f"   🔄 Topic Mixing Threshold: {self.get('topic_mixing_threshold')}")
        print(f"   🗄️  Qdrant Host: {self.get('qdrant_host')}:{self.get('qdrant_port')}")
        print(f"   📁 Collection Name: {self.get('collection_name')}")
    
    def validate_model_compatibility(self, document_model: str) -> bool:
        """Check if document model matches current settings"""
        current_model = self.get('embedding_model')
        if document_model != current_model:
            print(f"❌ Model mismatch: Document uses '{document_model}', current setting is '{current_model}'")
            print(f"   Please update settings or reprocess documents with model '{current_model}'")
            return False
        return True
