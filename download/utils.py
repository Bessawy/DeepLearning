from tqdm import tqdm
import urllib.request

def download_url(url, filename):

    response = urllib.request.urlopen(url)
    total_size = int(response.headers['content-length'])
    
    with open(filename, 'wb') as f:
        for data in tqdm(iter(lambda: response.read(32768), b""), total=total_size // 32768, unit='KB'):
            f.write(data)