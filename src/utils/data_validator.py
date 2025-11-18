import json
import os
from typing import List, Dict, Any


class DataValidator:
    @staticmethod
    def is_document_data(filepath: str) -> bool:
        """Check if file contains valid document data with expected structure"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            if not isinstance(data, list):
                return False
            
            if len(data) == 0:
                return False
            
            # Check first document for required fields
            first_doc = data[0]
            required_fields = ['id']
            
            for field in required_fields:
                if field not in first_doc:
                    return False
            
            # Check if it has document metadata fields
            metadata_fields = ['DC.title', 'DC.creator', 'DC.description', 'DC.subject']
            has_metadata = any(field in first_doc for field in metadata_fields)
            
            return has_metadata
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return False
    
    @staticmethod
    def get_document_files(directory: str) -> List[str]:
        """Get all valid document data files in directory"""
        if not os.path.exists(directory):
            return []
        
        document_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                if DataValidator.is_document_data(filepath):
                    document_files.append(filepath)
        
        return document_files
    
    @staticmethod
    def get_file_info(filepath: str) -> Dict[str, Any]:
        """Get information about a document data file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            doc_count = len(data)
            docs_with_descriptions = sum(1 for doc in data if doc.get('DC.description') and doc['DC.description'][0])
            docs_with_titles = sum(1 for doc in data if doc.get('DC.title') and doc['DC.title'][0])
            
            return {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'document_count': doc_count,
                'documents_with_descriptions': docs_with_descriptions,
                'documents_with_titles': docs_with_titles,
                'file_size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2)
            }
        except Exception as e:
            return {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'error': str(e)
            }
