import itertools

def get_experience_pairs(cfg):
    return [(cfg, p[0], p[1]) for p in list(itertools.product(cfg['exps'],cfg['exps'])) if p[0]!=p[1]]