from config import get_cfg_impl
from src.eval import do_eval
from src.cluster import do_cluster
from src.ringkey import do_ringkey
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

    if 'out_name' not in cfg:
        out_name = f'{loc_exp}_{ref_exp}'
    else:
        out_name = cfg['out_name']

    if os.path.exists(os.path.join(cfg['out_dir'], f"{out_name}_v3_recalls.csv")):
        return

    # Datasets
    loc_dataset, ref_dataset \
        = Dataset(cfg, loc_exp), Dataset(cfg, ref_exp)

    # Build descriptors
    loc_ringkeys = do_ringkey(
        cfg, loc_dataset)
    ref_ringkeys = do_ringkey(
        cfg, ref_dataset)

    # Sequence-sequence differences
    distances = build_difference_matrix(
        cfg, loc_ringkeys, ref_ringkeys)

    plt.imshow(distances)
    plt.savefig(os.path.join(cfg['out_dir'], 
        f"{out_name}_v1_distances.png"))

    # Performance metrics
    rcs, positives = do_eval(cfg, loc_dataset, ref_dataset, distances)

    plt.imshow(positives)
    plt.savefig(os.path.join(cfg['out_dir'], 
        f"{out_name}_positives.png"))

    csv_file = os.path.join(cfg['out_dir'], f"{out_name}_v1_recalls.csv")
    df = pd.DataFrame.from_dict(rcs)
    df.to_csv(csv_file, header=list(rcs.keys()))


if __name__ == "__main__":
    list(mp.Pool(processes=32).imap(run, get_experience_pairs(cfg)))