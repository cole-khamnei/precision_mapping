import os
import numpy as np

from . import constants, cifti_tools, utils
from . import surface_mapping as sfm

# ----------------------------------------------------------------------------# 
# -              Load Functions For Precision Map Intermediates              -# 
# ----------------------------------------------------------------------------# 


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
    return cifti_tools.get_cortex_data(vertex_labels.reshape(1, -1), cifti)[0]


def load_partition_labels(partition_path, template_cifti):
    """ """
    partition = np.load(partition_path)
    vertex_labels = get_partition_cortex(partition, template_cifti)
    vertex_labels[np.isnan(vertex_labels)] = np.nanmax(vertex_labels) + 1
    remapped_vertex_labels = np.unique(vertex_labels, return_inverse=True)[1]
    return remapped_vertex_labels


# ----------------------------------------------------------------------------# 
# -------------             Partition Dlabel Writers             -------------# 
# ----------------------------------------------------------------------------# 


def write_dlabel_precision_map(precision_map_values, save_path, label="", **kwargs):
    """ """
    precision_map_labels = precision_map_values.copy()
    precision_map_labels["left"] = precision_map_labels["left"].astype(str)
    precision_map_labels["right"] = precision_map_labels["right"].astype(str)
    sfm.write_labels_to_dlabel(precision_map_labels, save_path, label_name=label, **kwargs)


def write_parcel_dlabel(parcel_partition_path, parcel_dlabel_path, template_cifti,
                        pbar=True, cmap=None, overwrite=False):
    """ """
    args = [parcel_partition_path, parcel_dlabel_path]
    kwargs = dict(overwrite=overwrite, template_cifti=template_cifti,
                  pbar=pbar, cmap=cmap,
                  pbar_kwargs=dict(desc="Writing parcel dlabel"))
    if utils.multicall(write_network_dlabel, *args, **kwargs):
        return

    if os.path.exists(parcel_dlabel_path) and not overwrite:
        utils.printer(f"{parcel_dlabel_path} already exists, skipping. Can use --overwrite to overwrite.")
        return

    template_cifti = cifti_tools.get_template_cifti(template_cifti)
    vertex_parcel_labels = load_partition_labels(parcel_partition_path, template_cifti)

    parcel_values_int = cifti_tools.cifti_map(None, vertex_parcel_labels, template_cifti)
    write_dlabel_precision_map(parcel_values_int, parcel_dlabel_path, cmap=cmap)


def write_network_dlabel(network_partition_path, network_dlabel_path, template_cifti,
                         pbar=True, cmap=constants.NETWORK_CMAP, overwrite=False):
    """ """
    args = [network_partition_path, network_dlabel_path]
    kwargs = dict(overwrite=overwrite, template_cifti=template_cifti, cmap=cmap,
                  pbar=pbar, pbar_kwargs=dict(desc="Writing network dlabel"))
    if utils.multicall(write_network_dlabel, *args, **kwargs):
        return

    if os.path.exists(network_dlabel_path) and not overwrite:
        utils.printer(f"{network_dlabel_path} already exists, skipping. Can use --overwrite to overwrite.")
        return

    template_cifti = cifti_tools.get_template_cifti(template_cifti)
    vertex_network_labels, vertex_network_strings = np.load(network_partition_path)
    vertex_network_labels = vertex_network_labels.astype(int)
    network_values_int = cifti_tools.cifti_map(None, vertex_network_labels, template_cifti)
    network_name_to_num_map = {z: vertex_network_strings[np.where(vertex_network_labels == z)[0][0]]
                               for z in np.unique(vertex_network_labels)}
    network_dlabel_values = {
        "left": np.array(list(map(network_name_to_num_map.get, network_values_int["left"]))),
        "right": np.array(list(map(network_name_to_num_map.get, network_values_int["right"])))
    }
    write_dlabel_precision_map(network_dlabel_values, network_dlabel_path, cmap=cmap)


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
