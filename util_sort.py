import numpy as np
from util_gau import GaussianDataBasic

def gen_normal(gaus: GaussianDataBasic):
    # could substitute normal computation on-the-fly
    return gaus.normal


def presort_gaussian(gaus: GaussianDataBasic):
    pos = gaus.xyz
    center = np.mean(pos, axis=0)
    pos_recenter = pos - center[None]
    
    normal = gen_normal(gaus)
    
    normal_all = np.concatenate([normal, -normal])  # first len(pos) are original, second len(pos) are copys
    pos_recenter = np.concatenate([pos_recenter, pos_recenter])
    dot = (normal_all * pos_recenter).sum(axis=-1)
    index = np.argsort(dot)
    index = index % len(pos)
    
    normal = np.ascontiguousarray(normal_all).astype(np.float32).reshape(-1, 3)
    index = index.astype(np.int32).reshape(-1, 1)
    return normal, index
