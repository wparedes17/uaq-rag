import json
import os
from typing import List, Dict, Any


class FileManager:
    @staticmethod
    def ensure_directory(directory: str) -> None:
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    
    @staticmethod
    def save_json(data: Any, filepath: str) -> None:
        FileManager.ensure_directory(os.path.dirname(filepath))
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=2)
    
    @staticmethod
    def load_json(filepath: str) -> Any:
        with open(filepath) as file:
            return json.load(file)
    
    @staticmethod
    def list_files(directory: str, extension: str = None) -> List[str]:
        files = []
        for file in os.listdir(directory):
            if extension is None or file.endswith(extension):
                files.append(os.path.join(directory, file))
        return files
