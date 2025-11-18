#!/usr/bin/env python3
"""
UAQ RAG System Terminal Interface
A modular system for crawling documents and computing embeddings
"""

import sys
import os
import json
import argparse
from typing import List, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.crawlers.uaq_crawler import UAQCrawler
from src.core.embeddings import EmbeddingFactory
from src.core.vector_processor import VectorProcessor
from src.utils.data_validator import DataValidator
from src.utils.settings import Settings
from src.utils.file_manager import FileManager
from src.downloaders.file_downloader import FileDownloader
from src.core.http_client import Urllib3HttpClient

# Optional topic manager import
try:
    from src.core.topic_manager import TopicManager
    TOPIC_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Topic manager not available: {e}")
    print("   Install with: pip install qdrant-client scipy numpy")
    TOPIC_MANAGER_AVAILABLE = False


class RAGTerminalInterface:
    def __init__(self):
        self.crawler = None
        self.validator = DataValidator()
        self.settings = Settings()
        self.topic_manager = None
        self.file_downloader = FileDownloader(Urllib3HttpClient(), 'docs')
        
        # Initialize topic manager if available
        if TOPIC_MANAGER_AVAILABLE:
            try:
                self.topic_manager = TopicManager(self.settings)
            except ImportError as e:
                print(f"⚠️  Could not initialize topic manager: {e}")
                self.topic_manager = None
    
    def check_api_key(self, model: str) -> bool:
        """Check if required API key is available for the model"""
        if model.startswith("text-embedding"):
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or environment.")
                return False
        elif model.startswith("embed-"):
            api_key = os.getenv('COHERE_API_KEY')
            if not api_key:
                print("❌ Cohere API key not found. Please set COHERE_API_KEY in your .env file or environment.")
                return False
        elif model.startswith("huggingface/"):
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            # Hugging Face API key is optional for some models
            if api_key:
                print(f"✅ Using Hugging Face API key from environment")
            else:
                print(f"⚠️  No Hugging Face API key found - using public access (may have rate limits)")
        elif model.startswith("models/"):
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                print("❌ Google API key not found. Please set GOOGLE_API_KEY in your .env file or environment.")
                return False
        
        return True
    
    def run_crawler(self, pages: int = 77):
        """Run the crawler with specified number of pages"""
        print(f"🔍 Starting UAQ crawler with {pages} pages...")
        
        # Initialize crawler
        self.crawler = UAQCrawler()
        
        # Run crawling
        results = self.crawler.run_full_crawl(
            max_pages=pages,
            save_links=True,
            save_documents=True,
            download_files=False
        )
        
        print(f"\n✅ Crawling completed!")
        print(f"   Links found: {len(results['links'])}")
        print(f"   Documents parsed: {len(results['documents'])}")
        print(f"   Data saved to: data/raw/")
        
        return results
    
    def list_available_data(self):
        """List available document data files"""
        print("\n📁 Available document data files:")
        
        data_dir = "data/raw"
        if not os.path.exists(data_dir):
            print(f"   No data directory found at {data_dir}")
            return []
        
        document_files = self.validator.get_document_files(data_dir)
        
        if not document_files:
            print("   No valid document data files found")
            return []
        
        for i, filepath in enumerate(document_files, 1):
            info = self.validator.get_file_info(filepath)
            if 'error' in info:
                print(f"   {i}. {info['filename']} - ❌ Error: {info['error']}")
            else:
                print(f"   {i}. {info['filename']}")
                print(f"      📄 Documents: {info['document_count']}")
                print(f"      📝 With descriptions: {info['documents_with_descriptions']}")
                print(f"      📊 Size: {info['file_size_mb']} MB")
        
        return document_files
    
    def run(self):
        """Main terminal interface"""
        print("🎓 UAQ RAG System Terminal Interface")
        print("=" * 50)
        
        while True:
            print("\n📋 Main Menu:")
            print("   1. 🕷️  Run crawler")
            print("   2. 🧠 Compute embeddings")
            if self.topic_manager:
                print("   3. 🎯 Topic manager")
                print("   4. �  Swimming")
                print("   5. ⚙️  Settings")
                print("   6. 🚪 Exit")
                max_choice = 6
            else:
                print("   3. 🏊  Swimming")
                print("   4. ⚙️  Settings")
                print("   5. 🚪 Exit")
                max_choice = 5
            
            choice = input(f"\n🔢 Select option (1-{max_choice}): ").strip()
            
            if choice == '1':
                # Run crawler
                try:
                    pages = input("📄 Enter number of pages (default 77): ").strip()
                    pages = int(pages) if pages else 77
                    self.run_crawler(pages)
                except ValueError:
                    print("❌ Invalid number. Using default 77 pages.")
                    self.run_crawler(77)
            
            elif choice == '2':
                # Compute embeddings
                document_files = self.list_available_data()
                if not document_files:
                    continue
                
                selected_file = self.select_data_file(document_files)
                if not selected_file:
                    continue
                
                self.compute_embeddings(selected_file)
            
            elif choice == '3':
                if self.topic_manager:
                    # Topic manager
                    self.run_topic_manager()
                else:
                    # Swimming (when no topic manager)
                    self.run_swimming_menu()
            
            elif choice == '4':
                if self.topic_manager:
                    # Swimming
                    self.run_swimming_menu()
                else:
                    # Settings (when no topic manager)
                    self.run_settings_menu()
            
            elif choice == '5':
                if self.topic_manager:
                    # Settings
                    self.run_settings_menu()
                else:
                    # Exit (when no topic manager)
                    print("\n👋 Goodbye!")
                    break
            
            elif choice == '6' and self.topic_manager:
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid option. Please select 1-{}.".format(max_choice))
    
    def select_data_file(self, document_files: List[str]) -> str:
        """Let user select a data file"""
        if not document_files:
            return None
        
        while True:
            try:
                choice = input(f"\n🔢 Select file (1-{len(document_files)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(document_files):
                    return document_files[choice_idx]
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def run_topic_manager(self):
        """Topic management interface"""
        while True:
            print("\n🎯 Topic Manager:")
            print("   1. 🗄️  Create topic source store")
            print("   2. 📝 Create topic")
            print("   3. 🔄 Mix topics")
            print("   4. 📋 List user topics")
            print("   5. 📝 List mix topics")
            print("   6. ℹ️  Store info")
            print("   7. 🔙 Back to main menu")
            
            choice = input("\n🔢 Select option (1-7): ").strip()
            
            if choice == '1':
                self.topic_manager.create_topic_store()
            
            elif choice == '2':
                query = input("\n💭 Enter topic/query: ").strip()
                if query:
                    try:
                        limit = input("📄 Enter document limit (default: {}): ".format(
                            self.settings.get('default_topic_documents'))).strip()
                        limit = int(limit) if limit else None
                        self.topic_manager.create_topic(query, limit)
                    except ValueError:
                        print("❌ Invalid number. Using default limit.")
                        self.topic_manager.create_topic(query)
            
            elif choice == '3':
                try:
                    threshold = input("🔄 Enter similarity threshold (default: {}): ".format(
                        self.settings.get('topic_mixing_threshold'))).strip()
                    threshold = float(threshold) if threshold else None
                    self.topic_manager.mix_topics(threshold)
                except ValueError:
                    print("❌ Invalid number. Using default threshold.")
                    self.topic_manager.mix_topics()
            
            elif choice == '4':
                topics = self.topic_manager.list_user_topics()
                if topics:
                    print(f"\n📋 Found {len(topics)} user topics:")
                    for i, topic in enumerate(topics, 1):
                        print(f"   {i}. {topic['topic'][:60]}... (ID: {topic['topicID'][:8]}...)")
                        print(f"      📄 {topic.get('document_count', 0)} documents")
                        print(f"      📅 {topic.get('created_at', 'unknown')[:10]}")
                else:
                    print("ℹ️  No user topics found")
            
            elif choice == '5':
                mix_topics = self.topic_manager.list_mix_topics()
                if mix_topics:
                    print(f"\n📝 Found {len(mix_topics)} mix topics:")
                    for i, mix in enumerate(mix_topics, 1):
                        mix_name = mix.get('topic', 'Unnamed mix')
                        print(f"   {i}. {mix_name[:60]}... (ID: {mix.get('topicID', 'unknown')[:8]}...)")
                        print(f"      🔗 {mix.get('component_count', 0)} component topics")
                        print(f"      📄 {mix.get('document_count', 0)} total documents")
                        print(f"      📅 {mix.get('created_at', 'unknown')[:10]}")
                        print(f"      🎯 Similarity threshold: {mix.get('similarity_threshold', 'N/A')}")
                        
                        # Show component topics if available
                        component_topics = mix.get('component_topics', [])
                        if component_topics:
                            print(f"      📚 Component topic IDs: {', '.join([tid[:8] + '...' for tid in component_topics[:3]])}")
                            if len(component_topics) > 3:
                                print(f"         ... and {len(component_topics) - 3} more")
                else:
                    print("ℹ️  No mix topics found")
            
            elif choice == '6':
                store_info = self.topic_manager.get_store_info()
                print(f"\n📊 Topic Store Info:")
                print(f"   🗄️  Collection exists: {store_info.get('collection_exists', False)}")
                print(f"   📄 Vectors count: {store_info.get('vectors_count', 'N/A')}")
                print(f"   📊 Points count: {store_info.get('points_count', 'N/A')}")
                print(f"   📈 Status: {store_info.get('status', 'N/A')}")
                print(f"   🎯 User topics: {store_info.get('user_topics_count', 0)}")
                print(f"   🤖 Embedding model: {store_info.get('embedding_model', 'N/A')}")
            
            elif choice == '7':
                break
            
            else:
                print("❌ Invalid option. Please select 1-7.")
    
    def run_settings_menu(self):
        """Settings management interface"""
        while True:
            self.settings.display_settings()
            
            print("\n⚙️  Settings Menu:")
            print("   1. 🤖 Update embedding model")
            print("   2. 📄 Update default topic documents")
            print("   3. 🔄 Update topic mixing threshold")
            print("   4. 🔙 Back to main menu")
            
            choice = input("\n🔢 Select option (1-4): ").strip()
            
            if choice == '1':
                available_models = EmbeddingFactory.get_available_models()
                print("\n🤖 Available embedding models:")
                model_options = []
                
                for provider, models in available_models.items():
                    for model in models:
                        model_options.append(model)
                        print(f"   {len(model_options)}. {provider}: {model}")
                
                try:
                    choice = input(f"\n🔢 Select model (1-{len(model_options)}): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(model_options):
                        self.settings.update_embedding_model(model_options[choice_idx])
                    else:
                        print("❌ Invalid selection.")
                except ValueError:
                    print("❌ Please enter a valid number.")
            
            elif choice == '2':
                try:
                    count = input("📄 Enter default topic documents: ").strip()
                    count = int(count) if count else 5
                    if count > 0:
                        self.settings.update_topic_documents(count)
                    else:
                        print("❌ Number must be positive.")
                except ValueError:
                    print("❌ Please enter a valid number.")
            
            elif choice == '3':
                try:
                    threshold = input("🔄 Enter topic mixing threshold (0.0-1.0): ").strip()
                    threshold = float(threshold) if threshold else 0.85
                    if 0.0 <= threshold <= 1.0:
                        self.settings.update_mixing_threshold(threshold)
                    else:
                        print("❌ Threshold must be between 0.0 and 1.0.")
                except ValueError:
                    print("❌ Please enter a valid number.")
            
            elif choice == '4':
                break
            
            else:
                print("❌ Invalid option. Please select 1-4.")
    
    def compute_embeddings(self, input_file: str):
        """Compute embeddings for documents using settings model"""
        model = self.settings.get('embedding_model')
        print(f"\n🧠 Computing embeddings using {model}...")
        
        # Check API key availability
        if not self.check_api_key(model):
            return
        
        # Create embedding client
        try:
            embedding_client = EmbeddingFactory.create_embedding(model)
        except Exception as e:
            print(f"❌ Error creating embedding client: {e}")
            return
        
        # Create processor with settings
        processor = VectorProcessor(embedding_client, self.settings)
        
        # Generate output filename
        input_filename = os.path.basename(input_file)
        output_filename = input_filename.replace('.json', '_with_vectors.json')
        output_file = f"data/processed/{output_filename}"
        
        # Process documents
        try:
            result = processor.process_documents(input_file, output_file)
            
            print(f"\n✅ Embedding computation completed!")
            print(f"   Total documents: {result['total_documents']}")
            print(f"   Embeddings computed: {result['embeddings_computed']}")
            print(f"   Documents skipped: {result['documents_skipped']}")
            print(f"   Model used: {result['embedding_model']}")
            print(f"   Output saved to: {result['output_file']}")
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
    
    def run_swimming_menu(self):
        """Swimming interface for topic pool management"""
        while True:
            print("\n🏊 Swimming Menu:")
            print("   1. 📋 Create topic pool")
            print("   2. 🌊 Swimming")
            print("   3. 🔙 Back to main menu")
            
            choice = input("\n🔢 Select option (1-3): ").strip()
            
            if choice == '1':
                self.create_topic_pool()
            elif choice == '2':
                print("🌊 Swimming functionality not yet implemented")
            elif choice == '3':
                break
            else:
                print("❌ Invalid option. Please select 1-3.")
    
    def create_topic_pool(self):
        """Create topic pool by listing topics and downloading missing documents"""
        print("\n📋 Creating Topic Pool...")
        print("=" * 40)
        
        # Get topics from user and mix directories
        user_topics = self._get_topics_from_directory('topics/user')
        mix_topics = self._get_topics_from_directory('topics/mix')
        
        all_topics = user_topics + mix_topics
        
        if not all_topics:
            print("❌ No topics found in topics/user or topics/mix directories")
            return
        
        print(f"📊 Found {len(all_topics)} topics:")
        print(f"   📁 User topics: {len(user_topics)}")
        print(f"   📁 Mix topics: {len(mix_topics)}")
        
        # Collect all document IDs and PDF URLs
        all_documents = {}
        for topic_file, topic_data in all_topics:
            documents = topic_data.get('documents', [])
            for doc in documents:
                doc_id = str(doc.get('metadata', {}).get('id', ''))
                pdf_url = ''
                
                # Extract PDF URL from metadata
                citation_urls = doc.get('metadata', {}).get('citation_pdf_url', [])
                if citation_urls and len(citation_urls) > 0:
                    pdf_url = citation_urls[0]
                
                if doc_id and pdf_url:
                    all_documents[doc_id] = {
                        'pdf_url': pdf_url,
                        'title': doc.get('title', 'Unknown'),
                        'topic_file': topic_file
                    }
        
        print(f"\n📄 Found {len(all_documents)} unique documents")
        
        # Check which documents are already available
        missing_docs = self._check_missing_documents(all_documents)
        
        if not missing_docs:
            print("✅ All documents are already available in docs/ directory")
            return
        
        print(f"\n⬇️  Need to download {len(missing_docs)} missing documents")
        
        # Download missing documents
        downloaded_count = 0
        failed_count = 0
        
        for doc_id, doc_info in missing_docs.items():
            print(f"\n📥 Downloading document {doc_id}: {doc_info['title'][:50]}...")
            try:
                file_path = self.file_downloader.download_file(doc_info['pdf_url'], doc_id)
                print(f"✅ Downloaded to: {file_path}")
                downloaded_count += 1
            except Exception as e:
                print(f"❌ Failed to download: {e}")
                failed_count += 1
        
        print(f"\n🎉 Topic pool creation completed!")
        print(f"   ✅ Downloaded: {downloaded_count} documents")
        print(f"   ❌ Failed: {failed_count} documents")
        print(f"   📁 All documents saved in: docs/")
    
    def _get_topics_from_directory(self, directory: str) -> List[tuple]:
        """Get all topic files from a directory"""
        topics = []
        if not os.path.exists(directory):
            return topics
        
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        topic_data = json.load(f)
                    topics.append((filename, topic_data))
                except Exception as e:
                    print(f"⚠️  Error reading {filename}: {e}")
        
        return topics
    
    def _check_missing_documents(self, all_documents: dict) -> dict:
        """Check which documents are missing from docs/ directory"""
        missing = {}
        
        for doc_id, doc_info in all_documents.items():
            doc_path = os.path.join('docs', f'{doc_id}.pdf')
            if not os.path.exists(doc_path):
                missing[doc_id] = doc_info
        
        return missing


def main():
    parser = argparse.ArgumentParser(description='UAQ RAG System Terminal Interface')
    parser.add_argument('--pages', type=int, default=77, help='Number of pages to crawl (default: 77)')
    parser.add_argument('--mode', choices=['crawl', 'embeddings', 'interactive'], 
                       default='interactive', help='Run mode: crawl, embeddings, or interactive')
    
    args = parser.parse_args()
    
    interface = RAGTerminalInterface()
    
    if args.mode == 'crawl':
        interface.run_crawler(args.pages)
    elif args.mode == 'embeddings':
        document_files = interface.list_available_data()
        if document_files:
            selected_file = interface.select_data_file(document_files)
            if selected_file:
                interface.compute_embeddings(selected_file)
    else:
        interface.run()


if __name__ == "__main__":
    main()
