import re
from typing import List
from bs4 import BeautifulSoup
from ..core.http_client import HttpClientInterface


class LinkExtractor:
    def __init__(self, http_client: HttpClientInterface):
        self.http_client = http_client
        self.pattern_handle = re.compile(r'/handle/[0-9]+/[0-9]+')
    
    def extract_links_from_page(self, page_number: int, existing_links: List[str] = None) -> List[str]:
        if existing_links is None:
            existing_links = []
        
        url = f'https://ri-ng.uaq.mx/simple-search?query=&sort_by=score&order=desc&rpp=100&etal=0&start={page_number * 100}'
        html_content = self.http_client.get(url)
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for link in soup.find_all('a'):
            href = link.get('href')
            if href is not None and self.pattern_handle.match(href):
                existing_links.append(href)
        
        return existing_links
    
    def remove_duplicates(self, links: List[str]) -> List[str]:
        return list(dict.fromkeys(links))
