import os
from abc import ABC, abstractmethod
from typing import List, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional


class EmbeddingInterface(ABC):
    @abstractmethod
    def compute_embedding(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def compute_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass


class LiteLLMEmbedding(EmbeddingInterface):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        
        try:
            from litellm import embedding
            self.embedding = embedding
        except ImportError:
            raise ImportError("litellm is required. Install with: pip install litellm")
    
    def compute_embedding(self, text: str) -> List[float]:
        kwargs = {"model": self.model, "input": [text]}
        if self.api_key:
            kwargs["api_key"] = self.api_key
            
        response = self.embedding(**kwargs)
        return response.data[0]['embedding']
    
    def compute_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self.compute_embedding(text)
            embeddings.append(embedding)
        return embeddings


class EmbeddingFactory:
    @staticmethod
    def create_embedding(model: str, api_key: Optional[str] = None) -> EmbeddingInterface:
        return LiteLLMEmbedding(model=model, api_key=api_key)
    
    @staticmethod
    def get_available_models() -> dict:
        return {
            "OpenAI": [
                "text-embedding-3-small", 
                "text-embedding-3-large",
                "text-embedding-ada-002"
            ],
            "Cohere": [
                "embed-english-v3.0",
                "embed-multilingual-v3.0"
            ],
            "Hugging Face": [
                "huggingface/sentence-transformers/all-MiniLM-L6-v2"
            ],
            "Google": [
                "models/text-embedding-004"
            ]
        }
