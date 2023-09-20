from tqdm import tqdm
from time import sleep

with tqdm(total=100) as pbar:
    for i in range(10):
        sleep(0.1)
        pbar.update(1)