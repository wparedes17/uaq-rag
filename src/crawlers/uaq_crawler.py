import json
from typing import List, Dict, Tuple
from ..core.http_client import HttpClientInterface, Urllib3HttpClient
from .link_extractor import LinkExtractor
from ..parsers.document_parser import DocumentParser
from ..downloaders.file_downloader import FileDownloader


class UAQCrawler:
    def __init__(self, http_client: HttpClientInterface = None):
        self.http_client = http_client or Urllib3HttpClient()
        self.link_extractor = LinkExtractor(self.http_client)
        self.document_parser = DocumentParser(self.http_client)
        self.file_downloader = FileDownloader(self.http_client)
    
    def crawl_links(self, max_pages: int = 77) -> List[str]:
        links = []
        for page in range(max_pages):
            links = self.link_extractor.extract_links_from_page(page, links)
            print(f'Page {page + 1} done')
        
        return self.link_extractor.remove_duplicates(links)
    
    def parse_documents(self, links: List[str]) -> List[Dict[str, List[str]]]:
        documents_data = []
        for link in links:
            print(f'Document {link} started')
            document_data = self.document_parser.parse_document(link)
            documents_data.append(document_data)
            print(f'Document {link} done')
        
        return documents_data
    
    def get_download_links(self, documents_data: List[Dict[str, List[str]]]) -> List[Tuple[str, str]]:
        download_links = []
        for document in documents_data:
            if len(document['DC.title']) > 0 and document.get('citation_pdf_url'):
                download_link = document['citation_pdf_url'][0]
                filename = document['id']
                download_links.append((download_link, filename))
        
        return download_links
    
    def download_documents(self, documents_data: List[Dict[str, List[str]]]) -> List[str]:
        download_links = self.get_download_links(documents_data)
        return self.file_downloader.download_files_from_links(download_links)
    
    def save_to_json(self, data: List, filename: str) -> None:
        with open(filename, 'w') as file:
            json.dump(data, file)
    
    def load_from_json(self, filename: str) -> List:
        with open(filename) as file:
            return json.load(file)
    
    def run_full_crawl(self, max_pages: int = 77, save_links: bool = True, 
                      save_documents: bool = True, download_files: bool = True) -> Dict[str, any]:
        results = {}
        
        links = self.crawl_links(max_pages)
        results['links'] = links
        
        if save_links:
            self.save_to_json(links, 'data/raw/links.json')
        
        documents_data = self.parse_documents(links)
        results['documents'] = documents_data
        
        if save_documents:
            self.save_to_json(documents_data, 'data/raw/doctos_data.json')
        
        if download_files:
            downloaded_files = self.download_documents(documents_data)
            results['downloaded_files'] = downloaded_files
        
        return results
