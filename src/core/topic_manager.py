import json
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

# Optional imports for topic management
try:
    from .embeddings import EmbeddingFactory
    from .qdrant_store import QdrantTopicStore
    from ..utils.settings import Settings
    from ..utils.file_manager import FileManager
    TOPIC_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Topic management dependencies not available: {e}")
    print("   Install with: pip install qdrant-client scipy numpy")
    TOPIC_MANAGEMENT_AVAILABLE = False


class TopicManager:
    def __init__(self, settings: Settings = None):
        if not TOPIC_MANAGEMENT_AVAILABLE:
            raise ImportError("Topic management dependencies not installed. Run: pip install qdrant-client scipy numpy")
        
        self.settings = settings or Settings()
        self.qdrant_store = QdrantTopicStore(self.settings)
        self.embedding_model = self.settings.get('embedding_model')
        
        # Ensure topics directories exist
        os.makedirs('topics/user', exist_ok=True)
        os.makedirs('topics/mix', exist_ok=True)
    
    def _get_vector_files(self) -> List[str]:
        """Get list of files with vectors"""
        processed_dir = "data/processed"
        if not os.path.exists(processed_dir):
            return []
        
        files = []
        for file in os.listdir(processed_dir):
            if file.endswith('_with_vectors.json'):
                files.append(os.path.join(processed_dir, file))
        
        return sorted(files)
    
    def create_topic_store(self) -> bool:
        """Create or restore topic store from snapshot or documents"""
        print("🗄️  Creating Topic Store...")
        
        # List available options
        snapshots = self.qdrant_store.list_snapshots()
        document_files = self._get_vector_files()
        
        if not snapshots and not document_files:
            print("❌ No snapshots or document files found")
            return False
        
        # Show available options
        if snapshots:
            print("\n📋 Available snapshots:")
            for i, snap in enumerate(snapshots, 1):
                print(f"   {i}. {snap['display']}")
        
        if document_files:
            print("\n📄 Available document files:")
            for i, file in enumerate(document_files, len(snapshots) + 1):
                print(f"   {i}. {os.path.basename(file)}")
        
        # Build the prompt text
        if snapshots and document_files:
            prompt = f"\n🔢 Select option (1-{len(snapshots) + len(document_files)}) or 'n' for new: "
        elif snapshots:
            prompt = f"\n🔢 Select snapshot (1-{len(snapshots)}) or 'n' for new: "
        else:
            prompt = f"\n🔢 Select document file (1-{len(document_files)}): "
        
        while True:
            try:
                choice = input(prompt).strip()
                
                if choice.lower() == 'n' and document_files:
                    # Create new from documents
                    return self._create_new_store(document_files)
                
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(snapshots):
                    # Use snapshot
                    selected_snapshot = snapshots[choice_idx]
                    print(f"📂 Using snapshot: {selected_snapshot['display']}")
                    return self.qdrant_store.create_collection_from_snapshot(selected_snapshot)
                elif len(snapshots) <= choice_idx < len(snapshots) + len(document_files):
                    # Use document file
                    file_idx = choice_idx - len(snapshots)
                    selected_file = document_files[file_idx]
                    print(f"📂 Using document file: {os.path.basename(selected_file)}")
                    return self.qdrant_store.create_collection_from_documents(selected_file)
                elif not snapshots and 0 <= choice_idx < len(document_files):
                    # Only document files available
                    selected_file = document_files[choice_idx]
                    print(f"📂 Using document file: {os.path.basename(selected_file)}")
                    return self.qdrant_store.create_collection_from_documents(selected_file)
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def _create_new_store(self, document_files: List[str]) -> bool:
        print("\n📄 Select document file with vectors:")
        
        # List available processed files
        processed_dir = "data/processed"
        if not os.path.exists(processed_dir):
            print("❌ No processed data directory found")
            return False
        
        document_files = []
        for file in os.listdir(processed_dir):
            if file.endswith('_with_vectors.json'):
                filepath = os.path.join(processed_dir, file)
                document_files.append(filepath)
        
        if not document_files:
            print("❌ No document files with vectors found")
            return False
        
        for i, filepath in enumerate(document_files, 1):
            filename = os.path.basename(filepath)
            print(f"   {i}. {filename}")
        
        try:
            choice = int(input(f"\n🔢 Select file (1-{len(document_files)}): ").strip()) - 1
            if 0 <= choice < len(document_files):
                selected_file = document_files[choice]
                print(f"📂 Using document file: {os.path.basename(selected_file)}")
                return self.qdrant_store.create_collection_from_documents(selected_file)
            else:
                print("❌ Invalid selection")
                return False
        except ValueError:
            print("❌ Invalid input")
            return False
    
    def create_topic(self, query: str, document_limit: int = None) -> Optional[str]:
        """Create a topic from user query"""
        if document_limit is None:
            document_limit = self.settings.get('default_topic_documents')
        
        print(f"\n🎯 Creating topic: '{query}'")
        print(f"📄 Document limit: {document_limit}")
        
        try:
            # Create embedding client with current model
            embedding_client = EmbeddingFactory.create_embedding(self.embedding_model)
            
            # Compute query embedding
            print("🧠 Computing query embedding...")
            query_vector = embedding_client.compute_embedding(query)
            
            # Perform hybrid search
            print("🔍 Searching for relevant documents...")
            documents = self.qdrant_store.hybrid_search(query_vector, limit=document_limit)
            
            if not documents:
                print("❌ No documents found")
                return None
            
            # Create topic data
            topic_data = {
                "topicID": str(uuid.uuid4()),
                "topic": query,
                "documents": documents,
                "type": "user topic",
                "model": self.embedding_model,
                "vector": query_vector,
                "created_at": datetime.now().isoformat(),
                "document_count": len(documents)
            }
            
            # Save topic
            topic_filename = f"topics/user/topic_{topic_data['topicID']}.json"
            FileManager.save_json(topic_data, topic_filename)
            
            print(f"✅ Topic created successfully!")
            print(f"   Topic ID: {topic_data['topicID']}")
            print(f"   Documents found: {len(documents)}")
            print(f"   Saved to: {topic_filename}")
            
            # Show top documents
            print(f"\n📋 Top {min(3, len(documents))} documents:")
            for i, doc in enumerate(documents[:3], 1):
                print(f"   {i}. {doc['title'][:80]}... (Score: {doc['combined_score']:.3f})")
            
            return topic_data['topicID']
            
        except Exception as e:
            print(f"❌ Error creating topic: {e}")
            return None
    
    def list_user_topics(self) -> List[Dict[str, Any]]:
        """List all user topics"""
        user_dir = "topics/user"
        if not os.path.exists(user_dir):
            return []
        
        topics = []
        for file in os.listdir(user_dir):
            if file.startswith('topic_') and file.endswith('.json'):
                filepath = os.path.join(user_dir, file)
                try:
                    topic_data = FileManager.load_json(filepath)
                    topics.append(topic_data)
                except Exception:
                    continue
        
        return sorted(topics, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def list_mix_topics(self) -> List[Dict[str, Any]]:
        """List all mix topics"""
        mix_dir = "topics/mix"
        if not os.path.exists(mix_dir):
            return []
        
        mix_topics = []
        for file in os.listdir(mix_dir):
            if file.startswith('mixed_') and file.endswith('.json'):
                filepath = os.path.join(mix_dir, file)
                try:
                    mix_data = FileManager.load_json(filepath)
                    mix_topics.append(mix_data)
                except Exception:
                    continue
        
        return sorted(mix_topics, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def load_topic_vectors(self) -> List[Dict[str, Any]]:
        """Load vectors from all user topics for similarity analysis"""
        topics = self.list_user_topics()
        vectors = []
        
        for topic in topics:
            if topic.get('vector') and topic.get('topicID'):
                vectors.append({
                    'topicID': topic['topicID'],
                    'topic': topic['topic'],
                    'vector': topic['vector'],
                    'created_at': topic.get('created_at', ''),
                    'document_count': topic.get('document_count', 0)
                })
        
        return vectors
    
    def compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            import numpy as np
            
            v1, v2 = np.array(vec1), np.array(vec2)
            
            # Handle zero vectors
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return np.dot(v1, v2) / (norm1 * norm2)
            
        except ImportError:
            # Fallback without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    def find_similar_topics(self, threshold: float = None) -> List[List[Dict[str, Any]]]:
        """Find similar topics using cosine similarity"""
        if threshold is None:
            threshold = self.settings.get('topic_mixing_threshold')
        
        vectors = self.load_topic_vectors()
        if len(vectors) < 2:
            return []
        
        # Compute similarity matrix
        similarity_matrix = []
        for i, vec1 in enumerate(vectors):
            similarities = []
            for j, vec2 in enumerate(vectors):
                if i == j:
                    similarities.append(1.0)
                else:
                    sim = self.compute_cosine_similarity(vec1['vector'], vec2['vector'])
                    similarities.append(sim)
            similarity_matrix.append(similarities)
        
        # Use hierarchical clustering to find groups
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convert similarity to distance
            distance_matrix = [[1.0 - sim for sim in row] for row in similarity_matrix]
            
            # Perform hierarchical clustering
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='average')
            
            # Find clusters at the given threshold
            distance_threshold = 1.0 - threshold
            cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
            
            # Group topics by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(vectors[i])
            
            # Filter clusters with more than one topic
            mixed_topics = [cluster for cluster in clusters.values() if len(cluster) > 1]
            
            return mixed_topics
            
        except ImportError:
            print("⚠️  scipy not available, using simple similarity grouping")
            # Fallback: simple similarity grouping
            similar_groups = []
            used_indices = set()
            
            for i, vec1 in enumerate(vectors):
                if i in used_indices:
                    continue
                
                group = [vec1]
                used_indices.add(i)
                
                for j, vec2 in enumerate(vectors):
                    if i != j and j not in used_indices:
                        sim = self.compute_cosine_similarity(vec1['vector'], vec2['vector'])
                        if sim >= threshold:
                            group.append(vec2)
                            used_indices.add(j)
                
                if len(group) > 1:
                    similar_groups.append(group)
            
            return similar_groups
    
    def mix_topics(self, threshold: float = None) -> Optional[str]:
        """Mix similar topics using hierarchical clustering"""
        if threshold is None:
            threshold = self.settings.get('topic_mixing_threshold')
        
        print(f"\n🔄 Mixing similar topics (threshold: {threshold})...")
        
        # Clean existing mixed topics
        self.clean_mixed_topics()
        
        # Find similar topic groups
        similar_groups = self.find_similar_topics(threshold)
        
        if not similar_groups:
            print("ℹ️  No similar topics found for mixing")
            return None
        
        mixed_topic_id = str(uuid.uuid4())
        
        for i, group in enumerate(similar_groups):
            # Create mixed topic
            all_documents = []
            all_topic_ids = []
            combined_topic = " + ".join([topic['topic'] for topic in group])
            
            for topic in group:
                # Load full topic data
                topic_file = f"topics/user/topic_{topic['topicID']}.json"
                if os.path.exists(topic_file):
                    full_topic = FileManager.load_json(topic_file)
                    all_documents.extend(full_topic.get('documents', []))
                    all_topic_ids.append(topic['topicID'])
            
            # Remove duplicate documents
            seen_ids = set()
            unique_documents = []
            for doc in all_documents:
                doc_id = doc.get('id')
                if doc_id not in seen_ids:
                    unique_documents.append(doc)
                    seen_ids.add(doc_id)
            
            # Create mixed topic data
            mixed_topic = {
                "topicID": mixed_topic_id,
                "topic": combined_topic,
                "documents": unique_documents,
                "type": "mixed topic",
                "model": self.embedding_model,
                "component_topics": all_topic_ids,
                "similarity_threshold": threshold,
                "created_at": datetime.now().isoformat(),
                "document_count": len(unique_documents),
                "component_count": len(group)
            }
            
            # Save mixed topic
            mix_filename = f"topics/mix/mixed_{mixed_topic_id}_{i+1}.json"
            FileManager.save_json(mixed_topic, mix_filename)
            
            print(f"✅ Mixed topic created:")
            print(f"   Combined: {combined_topic}")
            print(f"   Documents: {len(unique_documents)}")
            print(f"   Components: {len(group)} topics")
            print(f"   Saved to: {mix_filename}")
        
        return mixed_topic_id
    
    def clean_mixed_topics(self):
        """Clean existing mixed topics directory"""
        mix_dir = "topics/mix"
        if os.path.exists(mix_dir):
            for file in os.listdir(mix_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(mix_dir, file))
            print("🧹 Cleaned existing mixed topics")
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the topic store"""
        collection_info = self.qdrant_store.get_collection_info()
        
        info = {
            'collection_exists': collection_info is not None,
            'user_topics_count': len(self.list_user_topics()),
            'embedding_model': self.embedding_model
        }
        
        if collection_info:
            # Use available attributes from CollectionInfo
            try:
                info.update({
                    'points_count': getattr(collection_info, 'points_count', 'N/A'),
                    'status': getattr(collection_info, 'status', 'N/A'),
                    'vectors_count': getattr(collection_info, 'vectors_count', 'N/A'),
                    'indexed_vectors_count': getattr(collection_info, 'indexed_vectors_count', 'N/A')
                })
            except Exception as e:
                print(f"⚠️  Could not get full collection info: {e}")
                info.update({
                    'points_count': 'N/A',
                    'status': 'N/A',
                    'vectors_count': 'N/A',
                    'indexed_vectors_count': 'N/A'
                })
        
        return info
