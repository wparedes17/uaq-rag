import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SearchRequest
from qdrant_client.http.models import CollectionInfo
from ..utils.settings import Settings
from ..utils.file_manager import FileManager
import httpx


class QdrantTopicStore:
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.client = QdrantClient(
            host=self.settings.get('qdrant_host'),
            port=self.settings.get('qdrant_port')
        )
        self.collection_name = self.settings.get('collection_name')
    
    def list_snapshots(self) -> List[Dict[str, str]]:
        """List available snapshots in topics/store directory"""
        snapshots = []
        snapshot_dir = "topics/store"
        
        if not os.path.exists(snapshot_dir):
            print("📸 No snapshots directory found")
            return snapshots
        
        # List all snapshot files
        for file in os.listdir(snapshot_dir):
            file_path = os.path.join(snapshot_dir, file)
            
            if file.endswith('.snapshot'):
                # Native Qdrant snapshot files (actual binary snapshots)
                try:
                    file_size = os.path.getsize(file_path)
                    snapshots.append({
                        'type': 'native',
                        'name': file,
                        'display': f"Native: {file} (Size: {file_size:,} bytes)",
                        'path': file_path
                    })
                except OSError:
                    snapshots.append({
                        'type': 'native',
                        'name': file,
                        'display': f"Native: {file} (Size: unknown)",
                        'path': file_path
                    })
                    
            elif file.endswith('_metadata.json') or file.endswith('.json'):
                # All JSON files are metadata - don't show in snapshot list
                # These are just metadata files, not actual snapshots
                continue
        
        # Sort by type and name
        native_snapshots = [s for s in snapshots if s['type'] == 'native']
        
        print(f"📸 Found {len(native_snapshots)} native snapshot files")
        if not native_snapshots:
            print("💡 No native snapshots found. Create one from documents to enable full restoration.")
        
        return native_snapshots
    
    def create_collection_from_snapshot(self, snapshot_info: Dict[str, str]) -> bool:
        """Create collection from snapshot"""
        try:
            if snapshot_info['type'] == 'native':
                # Handle native Qdrant snapshot
                snapshot_path = snapshot_info['path']
                snapshot_name = snapshot_info['name']
                
                # Delete existing collection if it exists
                try:
                    self.client.get_collection(self.collection_name)
                    print(f"⚠️  Collection '{self.collection_name}' already exists")
                    choice = input("Delete existing collection and restore from snapshot? (y/N): ").strip().lower()
                    if choice != 'y':
                        return False
                    
                    self.client.delete_collection(self.collection_name)
                except Exception:
                    pass
                
                # Restore from snapshot file
                print(f"📸 Restoring from native snapshot: {snapshot_name}")
                try:
                    # Qdrant's snapshot restoration uses the upload endpoint
                    # The correct endpoint is POST /collections/{collection_name}/snapshots/upload
                    restore_url = f"http://{self.settings.get('qdrant_host')}:{self.settings.get('qdrant_port')}/collections/{self.collection_name}/snapshots/upload?priority=snapshot"
                    
                    with open(snapshot_path, 'rb') as snapshot_file:
                        files = {'snapshot': (snapshot_name, snapshot_file.read(), 'application/octet-stream')}
                        with httpx.Client() as client:
                            response = client.post(restore_url, files=files)
                            response.raise_for_status()
                    
                    print(f"✅ Collection restored from native snapshot")
                    return True
                            
                except Exception as e:
                    print(f"❌ Error restoring from snapshot file: {e}")
                    print("💡 Alternative: Use Qdrant CLI to restore snapshot")
                    print(f"   Snapshot file location: {snapshot_path}")
                    print(f"   CLI command: curl -X POST 'http://{self.settings.get('qdrant_host')}:{self.settings.get('qdrant_port')}/collections/{self.collection_name}/snapshots/upload?priority=snapshot' -F 'snapshot=@{snapshot_path}'")
                    return False
                
            else:
                # Handle local snapshot info (our current implementation)
                snapshot_path = snapshot_info['path']
                with open(snapshot_path, 'r') as f:
                    snapshot_data = json.load(f)
                
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
                        size=snapshot_data.get('vector_size', 1536),
                        distance=Distance.COSINE
                    ),
                    # Enable payload indexing for hybrid search
                    optimizers_config={
                        "default_segment_number": 2,
                        "max_segment_size": 20000,
                        "memmap_threshold": 50000
                    }
                )
                
                # Try to restore data from the original source file
                source_file = snapshot_data.get('source_file')
                if source_file and os.path.exists(source_file):
                    print(f"📄 Restoring data from source file: {source_file}")
                    try:
                        # Load and upload the original documents
                        with open(source_file, 'r') as f:
                            documents = json.load(f)
                        
                        # Upload documents in batches
                        batch_size = 100
                        points = []
                        
                        for i, doc in enumerate(documents):
                            if doc.get('DC.vector'):
                                # Extract metadata
                                metadata = {
                                    'id': doc.get('id'),
                                    'title': doc.get('title', ''),
                                    'description': doc.get('DC.description', [''])[0] if doc.get('DC.description') else '',
                                    'date': doc.get('DC.date', [''])[0] if doc.get('DC.date') else '',
                                    'creators': doc.get('DC.creator', []),
                                    'subjects': doc.get('DC.subject', []),
                                    'publisher': doc.get('DC.publisher', ''),
                                    'format': doc.get('DC.format', ''),
                                    'language': doc.get('DC.language', ''),
                                    'rights': doc.get('DC.rights', ''),
                                    'type': doc.get('DC.type', ''),
                                    'identifier': doc.get('DC.identifier', ''),
                                    'source': doc.get('DC.source', ''),
                                    'relation': doc.get('DC.relation', ''),
                                    'coverage': doc.get('DC.coverage', ''),
                                    'freshness_score': self._calculate_freshness_score(doc.get('DC.date', [''])[0] if doc.get('DC.date') else ''),
                                    'metadata': doc
                                }
                                
                                # Create point
                                point = PointStruct(
                                    id=doc.get('id', i),
                                    vector=doc['DC.vector'],
                                    payload=metadata
                                )
                                points.append(point)
                                
                                # Upload batch
                                if len(points) >= batch_size:
                                    self.client.upsert(
                                        collection_name=self.collection_name,
                                        points=points
                                    )
                                    print(f"   Uploaded batch of {len(points)} points")
                                    points = []
                        
                        # Upload remaining points
                        if points:
                            self.client.upsert(
                                collection_name=self.collection_name,
                                points=points
                            )
                            print(f"   Uploaded final batch of {len(points)} points")
                        
                        print(f"✅ Collection '{self.collection_name}' created and data restored from local snapshot")
                        return True
                        
                    except Exception as e:
                        print(f"⚠️  Could not restore data from source file: {e}")
                        print(f"✅ Collection '{self.collection_name}' created from snapshot info (empty)")
                        print("💡 You'll need to upload documents separately")
                        return True
                else:
                    print(f"✅ Collection '{self.collection_name}' created from snapshot info (empty)")
                    print("💡 Original source file not found - you'll need to upload documents separately")
                    return True
            
        except Exception as e:
            print(f"❌ Error creating collection from snapshot: {e}")
            return False
    
    def create_native_snapshot(self) -> bool:
        """Create a native Qdrant snapshot"""
        try:
            # Create native snapshot
            snapshot_info = self.client.create_snapshot(self.collection_name)
            print(f"✅ Native snapshot created: {snapshot_info.name}")
            print(f"   Location: {snapshot_info.location}")
            print(f"   Size: {snapshot_info.size} bytes")
            print(f"   Created at: {snapshot_info.created_at}")
            
            # Download the snapshot file to local storage
            snapshot_url = f"http://{self.settings.get('qdrant_host')}:{self.settings.get('qdrant_port')}/collections/{self.collection_name}/snapshots/{snapshot_info.name}"
            
            try:
                with httpx.Client() as client:
                    response = client.get(snapshot_url)
                    response.raise_for_status()
                
                # Save snapshot file locally
                os.makedirs('topics/store', exist_ok=True)
                local_snapshot_path = os.path.join('topics/store', snapshot_info.name)
                
                with open(local_snapshot_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"📥 Snapshot downloaded to: {local_snapshot_path}")
                
                # Also create a small metadata file for reference
                metadata = {
                    'created_at': datetime.now().isoformat(),
                    'snapshot_name': snapshot_info.name,
                    'snapshot_file': snapshot_info.name,
                    'document_count': getattr(self.get_collection_info(), 'points_count', 0),
                    'vector_size': 1536,  # Default, could be retrieved from collection config
                    'embedding_model': self.settings.get('embedding_model'),
                    'type': 'native_snapshot'
                }
                
                metadata_file = local_snapshot_path.replace('.snapshot', '_metadata.json')
                FileManager.save_json(metadata, metadata_file)
                print(f"📄 Metadata saved to: {metadata_file}")
                
                return True
                
            except Exception as e:
                print(f"⚠️  Could not download snapshot file: {e}")
                print(f"💡 Snapshot is available at: {snapshot_url}")
                return True  # Still return True since snapshot was created
            
        except Exception as e:
            print(f"❌ Error creating native snapshot: {e}")
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
                # Enable payload indexing for hybrid search
                optimizers_config={
                    "default_segment_number": 2,
                    "max_segment_size": 20000,
                    "memmap_threshold": 50000
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
            
            # Save snapshot info
            snapshot_data = {
                'created_at': datetime.now().isoformat(),
                'document_count': len(documents),
                'vector_size': vector_size,
                'embedding_model': documents[0].get('embedding_model', 'unknown') if documents else 'unknown',
                'source_file': documents_file
            }
            
            os.makedirs('topics/store', exist_ok=True)
            snapshot_file = f"topics/store/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            FileManager.save_json(snapshot_data, snapshot_file)
            
            print(f"📄 Snapshot info saved to: {snapshot_file}")
            
            # Create and download native snapshot
            print("📸 Creating native Qdrant snapshot...")
            try:
                snapshot_info = self.client.create_snapshot(self.collection_name)
                print(f"✅ Native snapshot created: {snapshot_info.name}")
                
                # Download the snapshot file
                snapshot_url = f"http://{self.settings.get('qdrant_host')}:{self.settings.get('qdrant_port')}/collections/{self.collection_name}/snapshots/{snapshot_info.name}"
                
                with httpx.Client() as client:
                    response = client.get(snapshot_url)
                    response.raise_for_status()
                
                # Save snapshot file locally
                local_snapshot_path = os.path.join('topics/store', snapshot_info.name)
                
                with open(local_snapshot_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"📥 Native snapshot downloaded to: {local_snapshot_path}")
                
                # Update the JSON metadata to reference the native snapshot
                snapshot_data['native_snapshot'] = snapshot_info.name
                snapshot_data['snapshot_file'] = snapshot_info.name
                snapshot_data['type'] = 'native_snapshot'
                FileManager.save_json(snapshot_data, snapshot_file)
                
            except Exception as e:
                print(f"⚠️  Could not create/download native snapshot: {e}")
                print("💡 Collection is ready, but native snapshot creation failed")
            
            print(f"✅ Collection created with {len(points)} documents")
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
            # Use direct HTTP API approach - most reliable
            import requests
            import json
            
            url = f"http://{self.settings.get('qdrant_host')}:{self.settings.get('qdrant_port')}/collections/{self.collection_name}/points/search"
            
            payload_data = {
                "vector": query_vector,
                "limit": limit,
                "with_payload": True,
                "with_vector": False
            }
            
            response = requests.post(url, json=payload_data)
            response.raise_for_status()
            
            search_data = response.json()
            results = []
            
            for hit in search_data.get('result', []):
                hit_payload = hit.get('payload', {})
                results.append({
                    'id': hit.get('id'),
                    'score': hit.get('score'),
                    'title': hit_payload.get('title', ''),
                    'description': hit_payload.get('description', ''),
                    'date': hit_payload.get('date', ''),
                    'freshness_score': hit_payload.get('freshness_score', 0.5),
                    'metadata': hit_payload.get('metadata', {}),
                    'combined_score': hit.get('score') * (0.7 + 0.3 * hit_payload.get('freshness_score', 0.5))
                })
            
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
