from src.core.http_client import Urllib3HttpClient
from src.downloaders.file_downloader import FileDownloader


def download_pdfs_from_links():
    # Example download links (url, filename)
    download_links = [
        ("https://example.com/file1.pdf", "document_1"),
        ("https://example.com/file2.pdf", "document_2"),
    ]
    
    # Initialize components
    http_client = Urllib3HttpClient()
    downloader = FileDownloader(http_client, download_dir="data/raw")
    
    # Download files
    downloaded_files = downloader.download_files_from_links(download_links)
    
    print(f"Downloaded {len(downloaded_files)} files:")
    for file_path in downloaded_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    download_pdfs_from_links()
