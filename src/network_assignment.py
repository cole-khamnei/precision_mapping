import os

import numpy as np
import nibabel as nb
import scipy

from tqdm.auto import tqdm

from . import constants, utils
from . import cifti_tools, partition_tools
from .utils import colored

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

    # TODO: Readjust the FC connections to use the thresholded data (specifically sparse matrices)?
    fc_corr = utils.np_corr(FC_priors.T, roi_mean_FCs)
    sp_corr = utils.np_corr(spatial_priors.T, roi_index_set * 1)

    assert not np.isnan(fc_corr).any()

    sp_fc_corr = sp_corr * fc_corr
    sp_fc_index = np.argmax(sp_fc_corr, axis=0)

    return sp_fc_index[remapped_vertex_labels], network_labels[sp_fc_index[remapped_vertex_labels]], sp_corr, fc_corr


def assign_networks(cifti_paths, partition_path, save_path, censor_file=None, overwrite=False):
    """ """
    cifti_paths = [cifti_paths] if isinstance(cifti_paths, str) else cifti_paths

    if os.path.exists(save_path) and not overwrite:
        utils.printer(f"{save_path} already exists and no '--overwrite' flag given. Skipping network assignment.")
        return

    template_cifti = nb.load(cifti_paths[0])
    full_vertex_data = cifti_tools.load_voxel_data(cifti_paths, censor_file=censor_file)
    vertex_data = cifti_tools.get_cortex_data(full_vertex_data, template_cifti)

    partition = np.load(partition_path)
    vertex_labels = partition_tools.get_partition_cortex(partition, template_cifti)

    FC, spatial, network_labels = load_priors()
    vn, vns, sp_corr, fc_corr = get_network_assignment_labels(vertex_labels, vertex_data, network_labels, spatial, FC)
    
    if save_path:
        np.save(save_path, [vn, vns])
        np.save(save_path.replace(".npy", "_corrs.npy"), [sp_corr, fc_corr])
    
    utils.printer("Created network assignments.")
    return vn, vns, sp_corr, fc_corr


def assign_networks_batch(cifti_paths, partition_paths, save_paths,
                          censor_files=None, n_cores=None, overwrite=False):
    """ """
    cifti_paths = utils.list_wrap(cifti_paths, str)
    partition_paths = utils.list_wrap(partition_paths, str)
    save_paths = utils.list_wrap(save_paths, str)
    censor_files = [None] * len(save_paths) if censor_files is None else utils.list_wrap(censor_files, str)

    assert len(partition_paths) == len(save_paths)
    assert len(cifti_paths) == len(save_paths)

    arg_sets = zip(cifti_paths, partition_paths, save_paths, censor_files)
    single_assign_func = lambda args: assign_networks(*args, overwrite=overwrite)

    results = []
    gen_desc = "Assigning parcellation networks"
    pbar = tqdm(total=len(save_paths), desc=colored(gen_desc, constants.MAIN_PBAR_COLOR), unit="cifti", colour=constants.MAIN_PBAR_COLOR)
    for args in arg_sets:
        sample_label = os.path.basename(args[1]).split("_parcel")[0]
        pbar.set_description(colored(gen_desc + f" ({sample_label})", constants.MAIN_PBAR_COLOR))
        result = single_assign_func(args)
        pbar.update(1)
        results.append(result)
    pbar.set_description(colored(gen_desc, constants.MAIN_PBAR_COLOR))
    pbar.close()
    return results


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
