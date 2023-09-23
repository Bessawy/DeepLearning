import os
import zipfile
from utils import download_url

DATADIR = 'data/omniglot/data'
os.makedirs(DATADIR, exist_ok=True)

urls = [
    ("https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true", "images_background.zip", "images_background"),
    ("https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true", "images_evaluation.zip", "images_evaluation")
]

for url, filename, foldername in urls:
    download_url(url, filename)
    source_folder = os.path.join(DATADIR, foldername)

    # Unzip the file into the defined data directory
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(source_folder)

    os.remove(filename)

