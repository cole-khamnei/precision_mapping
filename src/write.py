import sys

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
    
from . import constants, utils
from . import network_assignment as na
from . import surface_mapping as sfm

# ----------------------------------------------------------------------------# 
# ------------             Multiple Argument Helpers              ------------# 
# ----------------------------------------------------------------------------# 


def check_multiple_args(args, main_dtype=str):
    """ """
    if any(not isinstance(arg, main_dtype) for arg in args):
        assert all(len(arg) == len(args[0]) for arg in args[1:]), "arg lists not same length"
        return True

    return False


# ----------------------------------------------------------------------------# 
# -              Load Functions For Precision Map Intermediates              -# 
# ----------------------------------------------------------------------------# 


def load_FC_cortex(FC_path, template_cifti):
    """ """
    sc = scipy.sparse.load_npz(FC_path)
    cortex_index = na.get_cortex_data(np.arange(sc.shape[0]).reshape(1, -1), template_cifti)[0]
    return sc[cortex_index][:, cortex_index]


def load_partition_labels(partition_path, template_cifti):
    """ """
    partition = np.load(partition_path)
    vertex_labels = na.get_partition_cortex(partition, template_cifti)
    vertex_labels[np.isnan(vertex_labels)] = np.nanmax(vertex_labels) + 1
    remapped_vertex_labels = np.unique(vertex_labels, return_inverse=True)[1]
    return remapped_vertex_labels


def load_network_labels(network_path):
    """ """
    return np.load(network_save_path)[0].astype(int)


# ----------------------------------------------------------------------------# 
# ---------------            Dlabel Write Functions            ---------------# 
# ----------------------------------------------------------------------------# 


def write_dlabel_precision_map(precision_map_values, save_path, label="", **kwargs):
    """ """
    precision_map_labels = precision_map_values.copy()
    precision_map_labels["left"] = precision_map_labels["left"].astype(str)
    precision_map_labels["right"] = precision_map_labels["right"].astype(str)
    sfm.write_labels_to_dlabel(precision_map_labels, save_path, label_name=label, **kwargs)


def write_parcel_dlabel(parcel_partition_path, parcel_dlabel_path, template_cifti, cmap=None):
    """ """
    # list of paths version
    args = [parcel_partition_path, parcel_dlabel_path]
    if check_multiple_args(args, main_dtype=str):
        np.vectorize(write_parcel_dlabel)(*args, template_cifti=template_cifti, cmap=cmap)
        return

    template_cifti = utils.get_template_cifti(template_cifti)
    vertex_parcel_labels = load_partition_labels(parcel_partition_path, template_cifti)
    
    parcel_values_int = utils.cifti_map(None, vertex_parcel_labels, template_cifti)
    write_dlabel_precision_map(parcel_values_int, parcel_dlabel_path, cmap=cmap)


def write_network_dlabel(network_partition_path, network_dlabel_path, template_cifti, cmap=constants.NETWORK_CMAP):
    """ """
    # list of paths version
    args = [network_partition_path, network_dlabel_path]
    if check_multiple_args(args, main_dtype=str):
        np.vectorize(write_network_dlabel)(*args, template_cifti=template_cifti, cmap=cmap)
        return

    template_cifti = utils.get_template_cifti(template_cifti)
    # single path version
    vertex_network_labels, vertex_network_strings = np.load(network_partition_path)
    vertex_network_labels = vertex_network_labels.astype(int)
    network_values_int = utils.cifti_map(None, vertex_network_labels, template_cifti)
    network_name_to_num_map = {z: vertex_network_strings[np.where(vertex_network_labels == z)[0][0]]
                               for z in np.unique(vertex_network_labels)}
    network_dlabel_values = {
        "left": np.array(list(map(network_name_to_num_map.get, network_values_int["left"]))),
        "right": np.array(list(map(network_name_to_num_map.get, network_values_int["right"])))
    }
    write_dlabel_precision_map(network_dlabel_values, network_dlabel_path, cmap=cmap)


# ----------------------------------------------------------------------------# 
# --------------------               Plots                --------------------# 
# ----------------------------------------------------------------------------# 


def vertex_plot(values, template_cifti, ax=None, pclip=(None, None), **kwargs):
    """ """
    values = utils.cifti_map(None, values, template_cifti)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    return sfm.surface_plot(values, ax=ax, **kwargs)


def parcel_plot(parcel_partition_path, network_partition_path, sample_label, save_path, template_cifti, close=True):
    """ """

    args = [parcel_partition_path, network_partition_path, sample_label, save_path]
    if check_multiple_args(args, main_dtype=str):
        np.vectorize(parcel_plot)(*args, template_cifti=template_cifti, close=close)
        return

    template_cifti = utils.get_template_cifti(template_cifti)
    vertex_parcel_labels = load_partition_labels(parcel_partition_path, template_cifti)
    vertex_network_labels, _ = np.load(network_partition_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.tight_layout(h_pad=2)
    ax, _ = vertex_plot(vertex_parcel_labels, template_cifti, cmap=plt.cm.Spectral, outline=False, ax=axes[0])
    ax.set_title(f"{sample_label} Assigned Parcels")
    
    ax, _ = vertex_plot(vertex_network_labels, template_cifti, cmap=plt.cm.Spectral, ax=axes[1])
    ax.set_title(f"{sample_label} Assigned Networks")
    
    fig.savefig(save_path, bbox_inches='tight')
    
    if close:
        plt.close()

    return

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
