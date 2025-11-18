import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.models import CollectionInfo
from ..utils.settings import Settings
from ..utils.file_manager import FileManager


class QdrantTopicStore:
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.client = QdrantClient(
            host=self.settings.get('qdrant_host'),
            port=self.settings.get('qdrant_port')
        )
        self.collection_name = self.settings.get('collection_name')
    
    def list_snapshots(self) -> List[str]:
        """List available snapshots in topics/store directory"""
        snapshot_dir = "topics/store"
        if not os.path.exists(snapshot_dir):
            return []
        
        snapshots = []
        for file in os.listdir(snapshot_dir):
            if file.endswith('.snapshot'):
                snapshots.append(os.path.join(snapshot_dir, file))
        
        return sorted(snapshots)
    
    def create_collection_from_snapshot(self, snapshot_path: str) -> bool:
        """Create collection from snapshot"""
        try:
            # Load snapshot info
            with open(snapshot_path, 'r') as f:
                snapshot_info = json.load(f)
            
            # Check if collection already exists
            try:
                self.client.get_collection(self.collection_name)
                print(f"⚠️  Collection '{self.collection_name}' already exists")
                choice = input("Delete existing collection and recreate? (y/N): ").strip().lower()
                if choice != 'y':
                    return False
                
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass  # Collection doesn't exist, that's fine
            
            # Create collection with hybrid search configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=snapshot_info.get('vector_size', 1536),
                    distance=Distance.COSINE
                ),
                # Enable payload indexing for hybrid search
                optimizers_config={
                    "default_segment_number": 2,
                    "max_segment_size": 20000,
                    "memmap_threshold": 50000
                }
            )
            
            print(f"✅ Collection '{self.collection_name}' created from snapshot")
            return True
            
        except Exception as e:
            print(f"❌ Error creating collection from snapshot: {e}")
            return False
    
    def create_collection_from_documents(self, documents_file: str) -> bool:
        """Create collection and upload documents with vectors"""
        try:
            # Load and validate documents
            with open(documents_file, 'r') as f:
                documents = json.load(f)
            
            if not documents:
                print("❌ No documents found in file")
                return False
            
            # Check for vectors and model metadata
            first_doc = documents[0]
            if 'DC.vector' not in first_doc:
                print("❌ Documents do not contain vectors (DC.vector field)")
                return False
            
            vector_size = len(first_doc['DC.vector'])
            document_model = first_doc.get('embedding_model', 'unknown')
            
            # Validate model compatibility
            if not self.settings.validate_model_compatibility(document_model):
                return False
            
            # Delete existing collection if it exists
            try:
                self.client.get_collection(self.collection_name)
                print(f"⚠️  Collection '{self.collection_name}' already exists")
                choice = input("Delete existing collection and recreate? (y/N): ").strip().lower()
                if choice != 'y':
                    return False
                
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            
            # Create collection with hybrid search configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
                # Enable text search for hybrid capabilities
                text_index_params={
                    "tokenizer": "word",
                    "min_token_len": 2,
                    "max_token_len": 15,
                    "lowercase": True
                }
            )
            
            # Prepare points for upload
            points = []
            for i, doc in enumerate(documents):
                if doc.get('DC.vector'):
                    # Extract metadata
                    metadata = {
                        'id': doc.get('id'),
                        'DC.creator': doc.get('DC.creator', []),
                        'DC.subject': doc.get('DC.subject', []),
                        'DC.type': doc.get('DC.type', []),
                        'DC.identifier': doc.get('DC.identifier', []),
                        'citation_pdf_url': doc.get('citation_pdf_url', []),
                        'DC.date': doc.get('DC.date', [])
                    }
                    
                    # Extract search fields
                    title = doc.get('DC.title', [''])[0] if doc.get('DC.title') else ''
                    description = doc.get('DC.description', [''])[0] if doc.get('DC.description') else ''
                    date = doc.get('DC.date', [''])[0] if doc.get('DC.date') else ''
                    
                    # Parse date for freshness weighting (YYYY-MM format)
                    freshness_score = self._calculate_freshness(date)
                    
                    point = PointStruct(
                        id=i,
                        vector=doc['DC.vector'],
                        payload={
                            'title': title,
                            'description': description,
                            'date': date,
                            'freshness_score': freshness_score,
                            'metadata': metadata
                        }
                    )
                    points.append(point)
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f"   Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            # Create snapshot info
            snapshot_info = {
                'created_at': datetime.now().isoformat(),
                'document_count': len(points),
                'vector_size': vector_size,
                'embedding_model': document_model,
                'source_file': documents_file
            }
            
            # Save snapshot info
            os.makedirs('topics/store', exist_ok=True)
            snapshot_file = f"topics/store/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            FileManager.save_json(snapshot_info, snapshot_file)
            
            print(f"✅ Collection created with {len(points)} documents")
            print(f"📄 Snapshot info saved to: {snapshot_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating collection from documents: {e}")
            return False
    
    def _calculate_freshness(self, date_str: str) -> float:
        """Calculate freshness score from date string (YYYY-MM format)"""
        try:
            if not date_str or len(date_str) < 7:
                return 0.5  # Default score for unknown dates
            
            # Parse year and month
            year = int(date_str[:4])
            month = int(date_str[5:7]) if len(date_str) > 5 else 1
            
            # Calculate months since 2000 (baseline)
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            months_since = (current_year - year) * 12 + (current_month - month)
            
            # Normalize to 0-1 scale (newer = higher score)
            # Assuming max 240 months (20 years) as baseline
            freshness = max(0, 1 - (months_since / 240))
            return freshness
            
        except Exception:
            return 0.5  # Default score for invalid dates
    
    def hybrid_search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and text search"""
        try:
            # Perform vector search with freshness weighting
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=None,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=0.0
            )
            
            # Process results
            results = []
            for hit in search_result:
                payload = hit.payload
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'title': payload.get('title', ''),
                    'description': payload.get('description', ''),
                    'date': payload.get('date', ''),
                    'freshness_score': payload.get('freshness_score', 0.5),
                    'metadata': payload.get('metadata', {}),
                    'combined_score': hit.score * (0.7 + 0.3 * payload.get('freshness_score', 0.5))
                })
            
            # Sort by combined score
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"❌ Error during hybrid search: {e}")
            return []
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get information about the collection"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            return None
