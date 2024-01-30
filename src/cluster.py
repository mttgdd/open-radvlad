from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans 

def do_cluster(cfg, dataset):        
    
    descs = []
    for img in tqdm(
        dataset, total=len(dataset), desc=f"Clustering"):
        descs.extend(img.tolist())
    descs = np.array(descs).astype(np.float32)

    kmeans_dict = KMeans(
        n_clusters=cfg['num_clusters'],
        init='k-means++', tol=0.0001, n_init=1,
        verbose=1).fit(descs)
    
    return kmeans_dict