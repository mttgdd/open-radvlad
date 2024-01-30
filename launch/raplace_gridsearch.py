from config import get_cfg_impl
from launch.raplace import run

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from copy import copy

# Get options
cfg = get_cfg_impl('config/test.yaml')
cfg['polar'] = False
cfg['cartesian'] = True
cfg['fft'] = False

os.makedirs(cfg['out_dir'], exist_ok=True)

experience_pairs = []
for scale in np.arange(0.1, 0.5, 0.1):
    for cart_res, cart_pw in [(1.2717, 256), (1.2717/2, 512), (1.2712*2, 128), (1.2712/4, 1024)]:
        cfg_new = copy(cfg)
        cfg_new['scale'] = scale
        cfg_new['cart_res'] = cart_res
        cfg_new['cart_pw'] = cart_pw
        cfg_new['out_name'] = f'{scale:.1f}_{cart_res}_{cart_pw}'
        experience_pairs.append((cfg_new, cfg['exps'][0], cfg['exps'][1]))

list(mp.Pool(processes=16).imap(run, experience_pairs))