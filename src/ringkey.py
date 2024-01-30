from tqdm import tqdm
import numpy as np

def do_ringkey(cfg, dataset):    
    ring_keys = []
    for img in tqdm(dataset, total=len(dataset), desc=f"Collecting ring keys"):
        ring_key = np.mean(img, axis=0)
        ring_keys.append(ring_key)
    return np.array(ring_keys)