import json
import os
from typing import List, Dict, Any
from datetime import datetime
from .embeddings import EmbeddingInterface
from ..utils.file_manager import FileManager
from ..utils.settings import Settings


class VectorProcessor:
    def __init__(self, embedding_client: EmbeddingInterface, settings: Settings = None):
        self.embedding_client = embedding_client
        self.settings = settings or Settings()
    
    def process_documents(self, input_filepath: str, output_filepath: str) -> Dict[str, Any]:
        """Process documents and add embeddings with model metadata"""
        # Load original documents
        with open(input_filepath, 'r', encoding='utf-8') as file:
            documents = json.load(file)
        
        processed_docs = []
        embedding_count = 0
        skipped_count = 0
        
        # Get embedding model info
        embedding_info = self.get_embedding_info()
        model_name = embedding_info.get('model', 'unknown')
        
        for doc in documents:
            processed_doc = doc.copy()
            
            # Check if document has description
            descriptions = doc.get('DC.description', [])
            if descriptions and descriptions[0]:
                try:
                    # Compute embedding for the first description
                    description = descriptions[0]
                    embedding = self.embedding_client.compute_embedding(description)
                    processed_doc['DC.vector'] = embedding
                    # Add model metadata
                    processed_doc['embedding_model'] = model_name
                    processed_doc['embedding_created_at'] = datetime.now().isoformat()
                    embedding_count += 1
                except Exception as e:
                    print(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                    processed_doc['DC.vector'] = None
                    processed_doc['embedding_model'] = None
                    skipped_count += 1
            else:
                processed_doc['DC.vector'] = None
                processed_doc['embedding_model'] = None
                skipped_count += 1
            
            processed_docs.append(processed_doc)
        
        # Save processed documents
        FileManager.save_json(processed_docs, output_filepath)
        
        return {
            'total_documents': len(documents),
            'embeddings_computed': embedding_count,
            'documents_skipped': skipped_count,
            'output_file': output_filepath,
            'embedding_model': model_name
        }
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        if hasattr(self.embedding_client, 'model'):
            return {
                'model': self.embedding_client.model,
                'provider': 'litellm'
            }
        return {'provider': 'litellm'}
