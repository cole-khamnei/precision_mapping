import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import cifti_tools, utils
from . import surface_mapping as sfm
from . import partition_tools as pt

# ----------------------------------------------------------------------------# 
# --------------            Precision Mapping Plots             --------------# 
# ----------------------------------------------------------------------------# 


def QC_plots(parcel_partition_path, save_path, close=True, overwrite=False):
    """ """
    mc_kwargs = dict(overwrite=overwrite, close=close, pbar=True,
                     pbar_kwargs=dict(desc="Generating QC plots"))
    if utils.multicall(QC_plots, parcel_partition_path, save_path, **mc_kwargs):
        return

    if os.path.exists(save_path) and not overwrite:
        utils.printer(f"{save_path} already exists, skipping. Can use --overwrite to overwrite.")
        return

    index, groups = np.load(parcel_partition_path)
    unique_groups, counts = np.unique(groups, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.kdeplot(counts, ax=ax, bw_method=0.1,
                label=f"Total Communities Found: {len(counts)}\nMedian Size: {np.median(counts) // 1}\nMax Size: {np.max(counts)}")
    ax.set(xlabel="Cluster Vertices", title="Distribution of Infomap Cluster Size")
    ax.legend(title="")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)

    if close:
        plt.close()


def vertex_plot(values, template_cifti, ax=None, **kwargs):
    """ """
    values = cifti_tools.cifti_map(None, values, template_cifti)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    return sfm.surface_plot(values, ax=ax, **kwargs)


def parcel_plot(parcel_partition_path, network_partition_path, sample_label,
                save_path, template_cifti, close=True, overwrite=False):
    """ """
    args = [parcel_partition_path, network_partition_path, sample_label, save_path]
    mc_kwargs = dict(overwrite=overwrite, template_cifti=template_cifti, close=close, pbar=True,
                     pbar_kwargs=dict(desc="Generating parcel plots"))
    if utils.multicall(parcel_plot, *args, **mc_kwargs):
        return

    if os.path.exists(save_path) and not overwrite:
        utils.printer(f"{save_path} already exists, skipping. Can use --overwrite to overwrite.")
        return

    template_cifti = cifti_tools.get_template_cifti(template_cifti)
    vertex_parcel_labels = pt.load_partition_labels(parcel_partition_path, template_cifti)
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


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
