import os
import gc

import numpy as np
import multiprocess as mp
import scipy

from sklearn.cluster import KMeans
from infomap import Infomap
from tqdm.auto import tqdm

from . import utils, cifti_tools

# ----------------------------------------------------------------------------# 
# ----------------           Infomaps Parcellating            ----------------# 
# ----------------------------------------------------------------------------# 


def save_partition(indices, groups, save_path, fill_value=np.nan):
    """ """
    full_indices, full_groups = np.arange(91_282).astype(int), np.full(91_282, fill_value=fill_value)
    full_groups[indices] = groups
    np.save(save_path, [full_indices, full_groups])


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

    indices = np.array(list(partition.keys()))
    groups = np.array(list(partition.values()))


    if save_path:
        save_partition(indices, groups, save_path)
    
    return indices, groups


# \section kmeans parcellating


def kmeans_parcel_detection(dtseries_paths, parcel_save_paths, censor_files, 
                            overwrite, seed, n_parcels,
                            filter_subcortex=True):
    """ """

    dtseries_paths = utils.list_wrap(dtseries_paths, str)
    parcel_save_paths = utils.list_wrap(parcel_save_paths, str)
    censor_files = utils.list_wrap(censor_files, str)

    assert len(dtseries_paths) == len(parcel_save_paths)
    if censor_files is not None:
        assert len(censor_files) == len(parcel_save_paths)
    else:
        censor_files = [None] * len(parcel_save_paths)

    desc = "Running kmeans parcel detection ({})"
    pbar = tqdm(total=len(parcel_save_paths), desc=desc)
    for dtseries_path, parcel_save_path, censor_file in zip(dtseries_paths, parcel_save_paths, censor_files):
        if os.path.exists(parcel_save_path) and not overwrite:
            utils.printer(f"{parcel_save_path} already exists and no '--overwrite' flag given. Skipping parcel detection.")
            pbar.update(1)
            continue
        

        sample_label = parcel_save_path.split("/")[-1].split("_kms")[0]
        pbar.set_description(desc.format(sample_label))

        voxel_data = cifti_tools.load_voxel_data(dtseries_path, censor_file=censor_file)
        indices = np.arange(voxel_data.shape[1])

        if filter_subcortex:
            # TODO: identify correct subcortical masking, believe 59_412 is correct
            subcortex_indices = indices >= 59_412
            indices, voxel_data = indices[~subcortex_indices], voxel_data[:, ~subcortex_indices]

            kmeans_l = KMeans(n_clusters=n_parcels + 1, random_state=seed).fit(voxel_data.T[:29_696])
            kmeans_r = KMeans(n_clusters=n_parcels + 1, random_state=seed).fit(voxel_data.T[29_696:59_412])

            labels = np.hstack([kmeans_l.labels_, kmeans_r.labels_])
            save_partition(indices, labels, parcel_save_path)
        else:
            kmeans = KMeans(n_clusters=n_parcels + 1, random_state=seed, n_init="auto").fit(voxel_data.T)
            save_partition(indices, kmeans.labels_, parcel_save_path)
        pbar.update(1)

    pbar.close()


# \section generic parcellation methods


def parcel_detection_single(corr_matrix, save_path, n_reps=1, silent=True,
                            overwrite=False, seed=137, **kwargs):
    """ """
    if os.path.exists(save_path) and not overwrite:
        utils.printer(f"{save_path} already exists and no '--overwrite' flag given. Skipping parcel detection.")
        return
    
    sc = scipy.sparse.load_npz(corr_matrix)
    partition = infomap_parcellation(sc, save_path=save_path, silent=silent,
                                     num_trials=n_reps, seed=seed, **kwargs)
    gc.collect()
    return partition


def parcel_detection(corr_matrix, save_path, n_cores=None, silent=True,
                     seed=137, **parcellating_kwargs):
    """ """
    corr_matrices = utils.list_wrap(corr_matrix, str)
    save_paths = utils.list_wrap(save_path, str)

    assert len(corr_matrices) == len(save_paths)
    
    arg_sets = zip(corr_matrices, save_paths)
    single_parcel_func = lambda args: parcel_detection_single(args[0], args[1], silent=silent, **parcellating_kwargs)
    desc = "Running infomap parcel detection"

    results = [single_parcel_func(arg_set) for arg_set in tqdm(arg_sets, total=len(save_paths), desc=desc)]

    # TODO: debug multiprocess in jn?
    # n_cores = utils.get_n_cores(n_cores)
    # with mp.Pool(n_cores) as p:
    #     results = []
    #     for result in tqdm(p.imap(single_parcel_func, arg_sets), total=len(save_paths), desc=desc):
    #         results.append(result)

    return results
    

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
