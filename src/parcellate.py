import os
import gc

import numpy as np
import scipy
import multiprocess as mp

from infomap import Infomap
from tqdm.auto import tqdm

from . import utils

from .utils import printer

# ----------------------------------------------------------------------------# 
# ----------------           Infomaps Parcellating            ----------------# 
# ----------------------------------------------------------------------------# 


def infomap_parcellation(matrix, save_path=None, num_trials=1, **kwargs):
    """ """

    row_counts = np.array((matrix > 0).sum(axis=0)).ravel()
    col_counts = np.array((matrix > 0).sum(axis=1)).ravel()
    vertex_edge_frac = np.mean((row_counts + col_counts) > 0) 
    if vertex_edge_frac <= 0.95:
        printer(f"WARNING: reduced number of vertex connections. {vertex_edge_frac}")

    infomap = Infomap(two_level=True, num_trials=num_trials, **kwargs)
    for r_i, c_i in zip(*matrix.nonzero()):
        infomap.add_link(r_i, c_i, weight=matrix[r_i, c_i])

    infomap.run()
    partition = infomap.get_modules()

    index = np.array(list(partition.keys()))
    values = np.array(list(partition.values()))

    if save_path:
        np.save(save_path, [index, values])
        # printer(f"infomap {save_path} done.")
    
    return index, values


def batch_infomap_parcellation(matrices, save_paths, n_cores=None, **infomap_kwargs):
    """ """
    n_cores = utils.get_n_cores(n_cores)

    arg_sets = zip(matrices, save_paths)
    single_parcel_func = lambda args: infomap_parcellation(args[0], args[1], silent=True, **infomap_kwargs)

    with mp.Pool(n_cores) as p:
        results = p.map_async(single_parcel_func, arg_sets)
        results = results.get()

    return results


def parcel_detection_single(corr_matrix, save_path, n_reps=1, silent=True,
                            overwrite=False, seed=42, **kwargs):
    """ """
    # TODO: figure out how to make this accepting of mujltiple / if I want accepting of multiple

    if os.path.exists(save_path) and not overwrite:
        printer(f"{save_path} already exists and no '--overwrite' flag given. Skipping parcel detection.")
        return
    
    print(corr_matrix)
    sc = scipy.sparse.load_npz(corr_matrix)
    partition = infomap_parcellation(sc, save_path=save_path, silent=silent,
                                     num_trials=n_reps, seed=seed, **kwargs)
    gc.collect()
    return partition


def parcel_detection(corr_matrix, save_path, n_cores=None, silent=True, **parcellating_kwargs):
    """ """
    # TODO: figure out how to make this accepting of mujltiple / if I want accepting of multiple

    corr_matrices = corr_matrix if isinstance(corr_matrix, str) else corr_matrix
    save_paths = save_path if isinstance(save_path, str) else save_path
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
    
    # printer(f"Created infomap partition")

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
