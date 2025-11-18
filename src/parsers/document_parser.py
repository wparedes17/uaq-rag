from typing import Dict, List
from bs4 import BeautifulSoup
from ..core.http_client import HttpClientInterface


class DocumentParser:
    def __init__(self, http_client: HttpClientInterface):
        self.http_client = http_client
        self.attributes = ['DC.title', 'DC.creator', 'DC.date', 'DC.description', 
                          'DC.subject', 'DC.type', 'DC.identifier', 'citation_pdf_url']
    
    def parse_document(self, handle: str) -> Dict[str, List[str]]:
        url = f'https://ri-ng.uaq.mx{handle}'
        html_content = self.http_client.get(url)
        soup = BeautifulSoup(html_content, 'html.parser')
        
        document_id = handle.split('/')[-1]
        document_data = {'id': document_id}
        
        for attribute in self.attributes:
            document_data[attribute] = []
            meta_tags = soup.find_all('meta', attrs={'name': attribute})
            
            for meta_tag in meta_tags:
                content = meta_tag.get('content')
                document_data[attribute].append(content)
        
        return document_data
