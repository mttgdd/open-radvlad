from config import get_cfg_impl
from src.eval import do_eval
from src.raplace import do_sino_fft, max_circular_xcorr
from src.dataset import Dataset
from launch.utils import get_experience_pairs

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np
import multiprocessing as mp
import random

# Get options
cfg = get_cfg_impl('config/test.yaml')
cfg['polar'] = False
cfg['cartesian'] = True
cfg['fft'] = False

os.makedirs(cfg['out_dir'], exist_ok=True)

def run(args):

    cfg, loc_exp, ref_exp = args

    if 'out_name' not in cfg:
        out_name = f'{loc_exp}_{ref_exp}'
    else:
        out_name = cfg['out_name']

    if os.path.exists(os.path.join(cfg['out_dir'], f"{out_name}_v2_recalls.csv")):
        return

    # Datasets
    loc_dataset, ref_dataset \
        = Dataset(cfg, loc_exp), Dataset(cfg, ref_exp)

    # Build descriptors
    loc_sino_ffts = do_sino_fft(
        cfg, loc_dataset)
    ref_sino_ffts = do_sino_fft(
        cfg, ref_dataset)

    # Sequence-sequence differences

    def distance_cb(loc_idx, ref_idx, loc_feats, ref_feats):
        loc_idx, ref_idx = int(loc_idx[0]), int(ref_idx[0])
        loc_feat, ref_feat = loc_feats[loc_idx], ref_feats[ref_idx]
        return max_circular_xcorr(loc_feat, ref_feat)

    distances = pairwise_distances(
        np.array(range(len(loc_sino_ffts))).reshape(-1, 1),
        np.array(range(len(ref_sino_ffts))).reshape(-1, 1),
        n_jobs=16, metric=lambda loc_idx, ref_idx: distance_cb(
            loc_idx, ref_idx, loc_sino_ffts, ref_sino_ffts)
    )

    plt.imshow(distances)
    plt.savefig(os.path.join(cfg['out_dir'], 
        f"{out_name}_v2_distances.png"))

    # Performance metrics
    rcs, positives = do_eval(cfg, loc_dataset, ref_dataset, distances)

    plt.imshow(positives)
    plt.savefig(os.path.join(cfg['out_dir'], 
        f"{out_name}_positives.png"))

    csv_file = os.path.join(cfg['out_dir'], f"{out_name}_v2_recalls.csv")
    df = pd.DataFrame.from_dict(rcs)
    df.to_csv(csv_file, header=list(rcs.keys()))

if __name__ == "__main__":
    experience_pairs = get_experience_pairs(cfg)
    # experience_pairs = list(reversed(experience_pairs))
    list(mp.Pool(processes=32).imap(run, experience_pairs))