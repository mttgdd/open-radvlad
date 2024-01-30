from tqdm import tqdm
import numpy as np

def get_vlad(cfg, kmeans_dict, img):
    vlad = np.zeros([cfg['num_clusters'], img.shape[1]])
    cluster_ids = kmeans_dict.predict(img)
    for j in range(img.shape[0]):
        vlad[cluster_ids[j], :] += img[j, :] - kmeans_dict.cluster_centers_[cluster_ids[j], :]
    vlad = vlad.flatten()
    vlad = np.sign(vlad)*np.sqrt(np.abs(vlad))
    vlad = vlad/np.sqrt(np.dot(vlad,vlad))
    return vlad

def do_vlad(cfg, dataset, kmeans_dict):    
    vlads = []
    for img in tqdm(dataset, total=len(dataset), desc=f"VLAD"):
        vlads.append(get_vlad(cfg, kmeans_dict, img))
    return vlads