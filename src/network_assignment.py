import os

import multiprocess as mp
import numpy as np
import nibabel as nb
import scipy

from tqdm.auto import tqdm

from . import utils
from . import constatnts

# ----------------------------------------------------------------------------# 
# -----------              Network Assignment Helpers              -----------# 
# ----------------------------------------------------------------------------# 


def load_priors(priors_path = constants.NETWORK_PRIORS):
    """ """
    priors = scipy.io.loadmat(priors_path)
    FC, spatial, network_labels, _ = priors["Priors"][0, 0]
    network_labels = np.array([lab[0][0] for lab in network_labels])
    FC, spatial = FC.T, spatial.T
    return FC, spatial, network_labels


def get_cortex_data(full_data, cifti):
    """ """
    pax = cifti.header.get_axis(1)
    slice_LUT = {structure: sl for structure, sl,_  in pax.iter_structures()}
    cortex_data_L = full_data[:, slice_LUT["CIFTI_STRUCTURE_CORTEX_LEFT"]]
    cortex_data_R = full_data[:, slice_LUT["CIFTI_STRUCTURE_CORTEX_RIGHT"]]
    return np.hstack([cortex_data_L, cortex_data_R])


def process_partition(partition, min_voxels=0, filter_subcortex=True):
    """ """
    index, groups = partition
    if filter_subcortex:
        # TODO: fix these issues, subcortex masking may not be correct
        subcortex_index = index >= (32_492 * 2)
        index, groups = index[~subcortex_index], np.unique(groups[~subcortex_index], return_inverse=True)[1]

    groups = np.random.permutation(np.max(groups))[groups - 1]    
    unique_groups, counts = np.unique(groups, return_counts=True)
    count_filter = np.isin(groups, unique_groups[counts >= min_voxels])
    
    vertex_labels = np.full(np.max(index) + 1, fill_value=np.nan)
    vertex_labels[index[count_filter]] = groups[count_filter]
    
    return vertex_labels, (index, groups)


def get_partition_cortex(partition, cifti):
    """ """
    vertex_labels, partition = process_partition(partition)
    return get_cortex_data(vertex_labels.reshape(1, -1), cifti)[0]


# ----------------------------------------------------------------------------# 
# ---------               Network Assignment Functions               ---------# 
# ----------------------------------------------------------------------------# 


def get_network_assignment_labels(vertex_labels, vertex_data, network_labels, spatial_priors, FC_priors):
    """ """
    vertex_labels[np.isnan(vertex_labels)] = np.nanmax(vertex_labels) + 1

    remapped_vertex_labels = np.unique(vertex_labels, return_inverse=True)[1]
    
    cluster_labels = np.sort(np.unique(remapped_vertex_labels))
    roi_index_set = remapped_vertex_labels.reshape(-1, 1) == cluster_labels
    
    roi_mean_signals = np.array([vertex_data[:, ri].mean(axis=1) for ri in roi_index_set.T]).T
    roi_mean_FCs = utils.np_corr(vertex_data, roi_mean_signals)

    # TODO: Readjust the FC connections to use the thresholded data (specifically sparse matrices)
    fc_corr = utils.np_corr(FC_priors.T, roi_mean_FCs)
    sp_corr = utils.np_corr(spatial_priors.T, roi_index_set * 1)

    assert not np.isnan(fc_corr).any()

    sp_fc_corr = sp_corr * fc_corr
    sp_fc_index = np.argmax(sp_fc_corr, axis=0)

    return sp_fc_index[remapped_vertex_labels], network_labels[sp_fc_index[remapped_vertex_labels]], sp_corr, fc_corr


def assign_networks(cifti_paths, partition_path, save_path, overwrite=False):
    """ """
    cifti_paths = [cifti_paths] if isinstance(cifti_paths, str) else cifti_paths

    if os.path.exists(save_path) and not overwrite:
        utils.printer(f"{save_path} already exists and no '--overwrite' flag given. Skipping network assignment.")
        return

    template_cifti = nb.load(cifti_paths[0])
    full_vertex_data = utils.load_voxel_data(cifti_paths)
    vertex_data = get_cortex_data(full_vertex_data, template_cifti)

    partition = np.load(partition_path)
    vertex_labels = get_partition_cortex(partition, template_cifti)

    FC, spatial, network_labels = load_priors()
    vn, vns, sp_corr, fc_corr = get_network_assignment_labels(vertex_labels, vertex_data, network_labels, spatial, FC)
    
    if save_path:
        np.save(save_path, [vn, vns])
        np.save(save_path.replace(".npy", "_corrs.npy"), [sp_corr, fc_corr])
    
    utils.printer("Created network assignments.")
    return vn, vns, sp_corr, fc_corr


def assign_networks_batch(cifti_paths, partition_paths, save_paths, n_cores=None, overwrite=False):
    """ """

    assert len(partition_paths) == len(save_paths)
    assert len(cifti_paths) == len(save_paths)

    arg_sets = zip(cifti_paths, partition_paths, save_paths)
    single_assign_func = lambda args: assign_networks(*args, overwrite=overwrite)

    desc = "Assigning networks"

    results = []
    pbar = tqdm(total=len(save_paths))
    for args in arg_sets:
        result = single_assign_func(args)
        pbar.update(1)
        results.append(result)

    # n_cores = utils.get_n_cores(n_cores)
    # print(n_cores)
    # with mp.Pool(n_cores) as p:
    #     results = []
    #     for result in tqdm(p.imap(single_assign_func, arg_sets), total=len(save_paths), desc=desc):
    #         results.append(result)

    return results


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
