import scipy

def build_difference_matrix(cfg, loc_vecs, ref_vecs):
    distances = scipy.spatial.distance.cdist(
        loc_vecs, ref_vecs,
        metric='euclidean'
    )
    return distances