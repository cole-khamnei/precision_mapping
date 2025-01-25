import numpy as np
import scipy
import nibabel as nb

from . import utils


# ----------------------------------------------------------------------------# 
# -----------              Network Assignment Helpers              -----------# 
# ----------------------------------------------------------------------------# 


def load_priors():
    """ """
    priors_path = "/data/data7/network_control/projects/voxel_analysis/resources/priors.mat"

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
    # vertex_labels = np.full(np.max(partition[0]) + 1, fill_value=np.nan)
    # vertex_labels[partition[0]] = partition[1]
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
    
    # roi_mean_signals = np.array([vertex_data[:, ri].mean(axis=1) for ri in roi_index_set.T]).T
    # roi_mean_FCs = utils.np_corr(vertex_data, roi_mean_signals)
    # fc_corr = utils.np_corr(FC_priors.T, roi_mean_FCs)
    # sp_corr = utils.np_corr(spatial_priors.T, roi_index_set * 1)
    # sp_fc_corr = sp_corr * fc_corr
    # sp_fc_index = np.argmax(sp_fc_corr, axis=0)

    fc_corr = 1
    sp_corr = utils.np_corr(spatial_priors.T, roi_index_set * 1)
    sp_fc_index = np.argmax(sp_corr, axis=0)

    return sp_fc_index[remapped_vertex_labels], network_labels[sp_fc_index[remapped_vertex_labels]], sp_corr, fc_corr


def assign_networks(cifti_paths, partition_path, save_path, verbose=True):
    """ """

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
    
    if verbose:
        print("Created network assignments.")

    return vn, vns, sp_corr, fc_corr

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
