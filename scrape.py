import requests
from bs4 import BeautifulSoup
import os

class DataDownloader:
    def __init__(self, base_url, url, data_dir, prefixes):
        self.base_url = base_url
        self.url = url
        self.data_dir = data_dir
        self.prefixes = prefixes

    def download_files(self):
        r = requests.get(self.url)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = soup.find_all('a')

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        for link in links:
            href = link.get('href')
            if href is not None and href.endswith('.XPT'):
                file_name = href.split('/')[-1]
                prefix = file_name.split('_')[0]
                if prefix in self.prefixes:
                    prefix_dir = os.path.join(self.data_dir, prefix)
                    if not os.path.exists(prefix_dir):
                        os.makedirs(prefix_dir)

                    file_path = os.path.join(prefix_dir, file_name)
                    if not os.path.exists(file_path):
                        print(f"Downloading {file_name}...")
                        res = requests.get(self.base_url + href)
                        res.raise_for_status()
                        with open(file_path, 'wb') as f:
                            for chunk in res.iter_content(102400):
                                f.write(chunk)
                        print(f"{file_name} downloaded successfully.")
