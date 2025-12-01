import os
import gc

import numpy as np
import multiprocess as mp
import scipy

from infomap import Infomap
from tqdm.auto import tqdm

from . import utils

# ----------------------------------------------------------------------------# 
# ----------------           Infomaps Parcellating            ----------------# 
# ----------------------------------------------------------------------------# 


def infomap_parcellation(matrix, save_path=None, num_trials=1, **kwargs):
    """ """

    row_counts = np.array((matrix > 0).sum(axis=0)).ravel()
    col_counts = np.array((matrix > 0).sum(axis=1)).ravel()
    vertex_edge_frac = np.mean((row_counts + col_counts) > 0) 
    if vertex_edge_frac <= 0.95:
        utils.printer(f"WARNING: reduced number of vertex connections. {vertex_edge_frac}")

    infomap = Infomap(two_level=True, num_trials=num_trials, **kwargs)
    for r_i, c_i in zip(*matrix.nonzero()):
        infomap.add_link(r_i, c_i, weight=matrix[r_i, c_i])

    infomap.run()
    partition = infomap.get_modules()

    index = np.array(list(partition.keys()))
    values = np.array(list(partition.values()))

    if save_path:
        np.save(save_path, [index, values])
    
    return index, values


def parcel_detection_single(corr_matrix, save_path, n_reps=1, silent=True,
                            overwrite=False, seed=42, **kwargs):
    """ """
    if os.path.exists(save_path) and not overwrite:
        utils.printer(f"{save_path} already exists and no '--overwrite' flag given. Skipping parcel detection.")
        return
    
    sc = scipy.sparse.load_npz(corr_matrix)
    partition = infomap_parcellation(sc, save_path=save_path, silent=silent,
                                     num_trials=n_reps, seed=seed, **kwargs)
    gc.collect()
    return partition


def parcel_detection(corr_matrix, save_path, n_cores=None, silent=True, **parcellating_kwargs):
    """ """

    corr_matrices = utils.list_wrap(corr_matrix, str)
    save_paths = utils.list_wrap(save_path, str)

    assert len(corr_matrices) == len(save_paths)

    
    arg_sets = zip(corr_matrices, save_paths)
    single_parcel_func = lambda args: parcel_detection_single(args[0], args[1],
                                                              silent=silent, **parcellating_kwargs)

    desc = "Running infomap parcel detection"
    n_cores = utils.get_n_cores(n_cores)
    with mp.Pool(n_cores) as p:
        results = []
        for result in tqdm(p.imap(single_parcel_func, arg_sets), total=len(save_paths), desc=desc):
            results.append(result)

    return results
    

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
