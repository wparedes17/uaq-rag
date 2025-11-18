#!/usr/bin/env python3
"""
UAQ RAG System Terminal Interface
A modular system for crawling documents and computing embeddings
"""

import sys
import os
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
                print("   4. ⚙️  Settings")
                print("   5. 🚪 Exit")
                max_choice = 5
            else:
                print("   3. ⚙️  Settings")
                print("   4. 🚪 Exit")
                max_choice = 4
            
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
                    # Settings (when no topic manager)
                    self.run_settings_menu()
            
            elif choice == '4':
                if self.topic_manager:
                    # Settings
                    self.run_settings_menu()
                else:
                    # Exit (when no topic manager)
                    print("\n👋 Goodbye!")
                    break
            
            elif choice == '5' and self.topic_manager:
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
            print("   5. ℹ️  Store info")
            print("   6. 🔙 Back to main menu")
            
            choice = input("\n🔢 Select option (1-6): ").strip()
            
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
                info = self.topic_manager.get_store_info()
                print("\n📊 Topic Store Info:")
                print(f"   🗄️  Collection exists: {info['collection_exists']}")
                if info['collection_exists']:
                    print(f"   📄 Vectors count: {info.get('vectors_count', 'N/A')}")
                    print(f"   📊 Points count: {info.get('points_count', 'N/A')}")
                    print(f"   📈 Status: {info.get('status', 'N/A')}")
                print(f"   🎯 User topics: {info['user_topics_count']}")
                print(f"   🤖 Embedding model: {info['embedding_model']}")
            
            elif choice == '6':
                break
            
            else:
                print("❌ Invalid option. Please select 1-6.")
    
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
