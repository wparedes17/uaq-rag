import os
from typing import List
from ..core.http_client import HttpClientInterface


class FileDownloader:
    def __init__(self, http_client: HttpClientInterface, download_dir: str = 'data/raw'):
        self.http_client = http_client
        self.download_dir = download_dir
        self._ensure_download_directory()
    
    def _ensure_download_directory(self):
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
    
    def download_file(self, url: str, filename: str) -> str:
        file_content = self.http_client.get(url)
        file_path = os.path.join(self.download_dir, f'{filename}.pdf')
        
        with open(file_path, 'wb') as file:
            file.write(file_content)
        
        return file_path
    
    def download_files_from_links(self, download_links: List[tuple]) -> List[str]:
        downloaded_files = []
        for url, filename in download_links:
            try:
                file_path = self.download_file(url, filename)
                downloaded_files.append(file_path)
                print(f'File {filename} downloaded successfully')
            except Exception as e:
                print(f'Failed to download {filename}: {e}')
        
        return downloaded_files
