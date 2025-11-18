from abc import ABC, abstractmethod
from typing import Any


class HttpClientInterface(ABC):
    @abstractmethod
    def get(self, url: str) -> Any:
        pass


class Urllib3HttpClient(HttpClientInterface):
    def __init__(self):
        import urllib3
        self.http = urllib3.PoolManager()
    
    def get(self, url: str) -> Any:
        response = self.http.request('GET', url)
        return response.data
