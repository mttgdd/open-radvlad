from config import get_cfg_impl
from src.eval import do_eval
from src.cluster import do_cluster
from src.vlad import do_vlad
from src.search import build_difference_matrix
from src.dataset import Dataset
from launch.utils import get_experience_pairs

import os
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

# Get options
cfg = get_cfg_impl('config/test.yaml')
cfg['polar'] = True
cfg['cartesian'] = False
cfg['fft'] = False

os.makedirs(cfg['out_dir'], exist_ok=True)

def run(args):

    cfg, loc_exp, ref_exp = args

    if os.path.exists(os.path.join(cfg['out_dir'], f"{loc_exp}_{ref_exp}_v3_recalls.csv")):
        return

    # Datasets
    loc_dataset, ref_dataset \
        = Dataset(cfg, loc_exp), Dataset(cfg, ref_exp)

    # Find cluster centres
    kmeans_dict = do_cluster(cfg, ref_dataset)

    # Build descriptors
    loc_vlads = do_vlad(
        cfg, loc_dataset, kmeans_dict)
    ref_vlads = do_vlad(
        cfg, ref_dataset, kmeans_dict)

    # Sequence-sequence differences
    distances = build_difference_matrix(
        cfg, loc_vlads, ref_vlads)

    plt.imshow(distances)
    plt.savefig(os.path.join(cfg['out_dir'], 
        f"{loc_exp}_{ref_exp}_v3_distances.png"))

    # Performance metrics
    rcs, positives = do_eval(cfg, loc_dataset, ref_dataset, distances)

    plt.imshow(positives)
    plt.savefig(os.path.join(cfg['out_dir'], 
        f"{loc_exp}_{ref_exp}_positives.png"))

    csv_file = os.path.join(cfg['out_dir'], f"{loc_exp}_{ref_exp}_v3_recalls.csv")
    df = pd.DataFrame.from_dict(rcs)
    df.to_csv(csv_file, header=list(rcs.keys()))

experience_pairs = get_experience_pairs(cfg)
experience_pairs = list(reversed(experience_pairs))
list(mp.Pool(processes=32).imap(run, experience_pairs))