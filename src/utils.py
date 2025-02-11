import shutil
import gc
import os

import numpy as np
import scipy
import nibabel as nb
import multiprocess as mp

from tqdm.auto import tqdm
from time import sleep

# \section random helpers

class Printer:
    def __init__(self, silent=False):
        self.silent = silent

    def __call__(self, *args):
        """ """
        if not self.silent:
            print(*args)

    def mute(self):
        self.silent = True

    def unmute(self):
        self.silent = False


printer = Printer(silent=False)


def cache_tmp_path(path, use_cache=True, write_cache=True):
    """ """
    if not isinstance(path, str):
        return [cache_tmp_path(path_i, use_cache=use_cache, write_cache=write_cache)
                for path_i in tqdm(path, desc="Caching paths in ~/_tmp")]

    if not use_cache:
        return path

    # TODO: builtin hash is non-deterministic, use hashlib or z-adler if want hashed option
    # tmp_path = f"/tmp/cifti_H{hash(path)}.{path.split('.', maxsplit=1)[1]}"

    tmp_path = "~/_tmp/" + path.split("/data/data", maxsplit=1)[1].lstrip("1234567/").replace("/", "--")
    tmp_path = os.path.expanduser(tmp_path)

    if not os.path.exists(tmp_path):
        if write_cache:
            shutil.copyfile(path, tmp_path)
            gc.collect()
        else:
            tmp_path = path

    return tmp_path

# ----------------------------------------------------------------------------# 
# --------------------             Np Helpers             --------------------# 
# ----------------------------------------------------------------------------# 


def np_corr(x, y):
    """ """
    x, y = x.T, y.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    y_demeaned = y - y.mean(axis=1, keepdims=True)

    x_norm = x_demeaned / np.sqrt(np.sum(x_demeaned ** 2, axis=1, keepdims=True))
    y_norm = y_demeaned / np.sqrt(np.sum(y_demeaned ** 2, axis=1, keepdims=True))
    return x_norm @ y_norm.T


# ----------------------------------------------------------------------------# 
# --------------------           Cifti Helpers            --------------------# 
# ----------------------------------------------------------------------------# 


def load_voxel_data(dtseries_paths):
    """ """
    if not isinstance(dtseries_paths, str):
        return np.vstack([load_voxel_data(path) for path in dtseries_paths])

    cifti = nb.load(dtseries_paths)
    sleep(5)
    return cifti.get_fdata()


# ----------------------------------------------------------------------------# 
# -----------------           Multiprocess Helpers           -----------------# 
# ----------------------------------------------------------------------------# 


def get_n_cores(n_cores=None, cpu_offset=1):
    """ """
    max_cpus = mp.cpu_count()
    if n_cores is None:
        return max_cpus - cpu_offset

    return min(max_cpus, n_cores)


# \section path helpers


def create_path_tag(prefix, sparsity, mask, exclude_subcortex, max_trs=None, dist_threshold=10):
    """ """
    
    subcortex_status = "_SC" if not exclude_subcortex else ""
    mask_tag = f"_D{dist_threshold}" if mask else ""
    max_trs_tag = f"_TR{max_trs:0.0f}" if max_trs else ""
    tag = f"S{sparsity * 10:.0f}{mask_tag}{subcortex_status}{max_trs_tag}"
    
    return f"{prefix}_{tag}"


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
