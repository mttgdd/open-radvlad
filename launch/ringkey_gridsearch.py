from config import get_cfg_impl
from launch.ringkey import run

import os
import multiprocessing as mp
from copy import copy

# Get options
cfg = get_cfg_impl('config/test.yaml')
cfg['polar'] = True
cfg['cartesian'] = False
cfg['fft'] = False

os.makedirs(cfg['out_dir'], exist_ok=True)

experience_pairs = []
for bin_dim in [128, 512]:
    for max_bin in [int(3768/2), 3768]:
        for num_azis in [50, 100, 200, 400]:
            cfg_new = copy(cfg)
            cfg_new['bin_dim'] = bin_dim
            cfg_new['max_bin'] = max_bin
            cfg_new['num_azis'] = num_azis
            cfg_new['out_name'] = f'{bin_dim}_{max_bin}_{num_azis}'
            experience_pairs.append((cfg_new, cfg['exps'][0], cfg['exps'][1]))

list(mp.Pool(processes=40).imap(run, experience_pairs))