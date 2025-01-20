import argparse
import os
import sys

import numpy as np
import scipy
import nibabel as nb
import matplotlib.pyplot as plt
import seaborn as sns

from infomap import Infomap

voxel_analysis_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(voxel_analysis_dir, "../")
sys.path.insert(0, project_path)

import torch_math_tools as tmt

import CAP_tools
from CAP_tools import sfm

# ----------------------------------------------------------------------------# 
# --------------------             Constants              --------------------# 
# ----------------------------------------------------------------------------# 

DIST_DIR = "/data/data7/network_control/projects/network_control/resources/brain_distances"
SUBCORTEX_MASK_PATH = os.path.join(DIST_DIR, "subcortex_mask.npy")
GEODESIC_MASK_PATH = os.path.join(DIST_DIR, f"geodesic_mask_30.npz")
GEODESIC_MASK_GENERIC_PATH = os.path.join(DIST_DIR, "geodesic_mask_{dist}.npz")
DIST_THRESHOLD = 30

# ----------------------------------------------------------------------------# 
# --------------------             Functions              --------------------# 
# ----------------------------------------------------------------------------# 


def get_cortex_data(full_data, cifti):
    pax = cifti.header.get_axis(1)
    slice_LUT = {structure: sl for structure, sl,_  in pax.iter_structures()}
    cortex_data_L = full_data[:, slice_LUT["CIFTI_STRUCTURE_CORTEX_LEFT"]]
    cortex_data_R = full_data[:, slice_LUT["CIFTI_STRUCTURE_CORTEX_RIGHT"]]
    return np.hstack([cortex_data_L, cortex_data_R])


def load_voxel_data(dtseries_paths):
    """ """
    if not isinstance(dtseries_paths, str):
        return np.vstack([load_voxel_data(path) for path in dtseries_paths])

    cifti = nb.load(dtseries_paths)
    return cifti.get_fdata()


def generate_voxel_FC(voxel_data, save_path=None, sparsity=0.1, exclude_index_path=None,
                      mask_path=None, block_size=5000, **SC_kwargs):
    """ """

    exclude_index = np.load(exclude_index_path) if exclude_index_path else None 
    mask = scipy.sparse.load_npz(mask_path) if mask_path else None

    # TODO: Fix masking related issues
    # TODO: add infomaps check, to insure that at least a certain percent of vertices have connections
    # mask = None
    sc = tmt.matrix.SparseCorrelator.run(voxel_data[:, :], mask=mask, symmetric=True,
                                         exclude_index=exclude_index,
                                         sparsity_percent=sparsity, block_size=block_size)
    if save_path:
        scipy.sparse.save_npz(save_path, sc)

    return sc


# ----------------------------------------------------------------------------# 
# ----------              Infomap Partition Functions               ----------# 
# ----------------------------------------------------------------------------# 


def infomap_parcellation(matrix, save_path=None, num_trials=1, **kwargs):
    """ """

    row_counts = np.array((matrix > 0).sum(axis=0)).ravel()
    col_counts = np.array((matrix > 0).sum(axis=1)).ravel()
    vertex_edge_frac = np.mean((row_counts + col_counts) > 0) 
    if vertex_edge_frac <= 0.95:
        print(f"WARNING: reduced number of vertex connections. {vertex_edge_frac}")

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


def process_partition(partition, cifti, min_voxels=0, n_vertices=91_282, filter_subcortex=True):
    """ """
    index, groups = partition
    if filter_subcortex:
        # TODO: fix these issues, subcortex masking may not be correct
        subcortex_index = index >= (32_492 * 2)
        index, groups = index[~subcortex_index], np.unique(groups[~subcortex_index], return_inverse=True)[1]

    groups = np.random.permutation(np.max(groups))[groups - 1]    
    unique_groups, counts = np.unique(groups, return_counts=True)
    count_filter = np.isin(groups, unique_groups[counts >= min_voxels])
    
    vertex_labels = np.full(n_vertices, fill_value=np.nan)
    vertex_labels[index[count_filter]] = groups[count_filter]
    
    precision_map_values = CAP_tools.utils.cifti_map(None, vertex_labels, cifti)
    return precision_map_values, (index, groups)


def get_partition_cortex(partition, cifti):
    """ """
    _, partition = process_partition(partition, cifti)
    vertex_labels = np.full(np.max(partition[0]) + 1, fill_value=np.nan)
    vertex_labels[partition[0]] = partition[1]
    return get_cortex_data(vertex_labels.reshape(1, -1), cifti)[0]


def np_corr(x, y):
    """ """
    x, y = x.T, y.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    y_demeaned = y - y.mean(axis=1, keepdims=True)

    x_norm = x_demeaned / np.sqrt(np.sum(x_demeaned ** 2, axis=1, keepdims=True))
    y_norm = y_demeaned / np.sqrt(np.sum(y_demeaned ** 2, axis=1, keepdims=True))
    return x_norm @ y_norm.T


def load_priors():
    """ """
    priors_path = "/data/data7/network_control/projects/voxel_analysis/resources/priors.mat"

    priors = scipy.io.loadmat(priors_path)
    FC, spatial, network_labels, _ = priors["Priors"][0, 0]
    network_labels = np.array([lab[0][0] for lab in network_labels])
    FC, spatial = FC.T, spatial.T
    return FC, spatial, network_labels


def cortex_plot(data, cifti, rois=None, cmap=plt.cm.coolwarm):
    """ """
    data_values = CAP_tools.utils.cifti_map(rois, data, cifti)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax, _ = sfm.surface_plot(data_values, cmap=cmap, ax=ax)
    return ax, data_values


def get_network_assigments(vertex_labels, vertex_data, network_labels, spatial_priors, FC_priors):
    """ """
    vertex_labels[np.isnan(vertex_labels)] = np.nanmax(vertex_labels) + 1

    remapped_vertex_labels = np.unique(vertex_labels, return_inverse=True)[1]
    
    cluster_labels = np.sort(np.unique(remapped_vertex_labels))
    roi_index_set = remapped_vertex_labels.reshape(-1, 1) == cluster_labels
    
    roi_mean_signals = np.array([vertex_data[:, ri].mean(axis=1) for ri in roi_index_set.T]).T
    roi_mean_FCs = np_corr(vertex_data, roi_mean_signals)
    fc_corr = np_corr(FC_priors.T, roi_mean_FCs)
    sp_corr = np_corr(spatial_priors.T, roi_index_set * 1)
    sp_fc_corr = sp_corr * fc_corr
    
    sp_fc_index = np.argmax(sp_fc_corr, axis=0)

    return sp_fc_index[remapped_vertex_labels], network_labels[sp_fc_index[remapped_vertex_labels]]

# ----------------------------------------------------------------------------# 
# --------------------               Plots                --------------------# 
# ----------------------------------------------------------------------------# 


def plot_precision_map(precision_map_values, title="", save_path=None):
    """ """
    n_parcels = max(np.nanmax(precision_map_values["left"]), np.nanmax(precision_map_values["right"]))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax, _ = sfm.surface_plot(precision_map_values, cmap=plt.cm.Spectral, ax=ax)
    ax.set_title(f"{title} Precision Map\nNumber of Parcels: {n_parcels:0.0f}")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)


def precision_map_QC_plots(partition, save_path=None):
    """ """
    index, groups = partition
    unique_groups, counts = np.unique(groups, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.kdeplot(counts, ax=ax, bw_method=0.1,
                label=f"Total Communities Found: {len(counts)}\nMedian Size: {np.median(counts) // 1}\nMax Size: {np.max(counts)}")
    ax.set(xlabel="Cluster Vertices", title="Distribution of Infomap Cluster Size")
    ax.legend(title="")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)


def write_dlabel_precision_map(precision_map_values, save_path, label=""):
    """ """
    precision_map_labels = precision_map_values.copy()
    precision_map_labels["left"] = precision_map_labels["left"].astype(str)
    precision_map_labels["right"] = precision_map_labels["right"].astype(str)
    CAP_tools.sfm.write_labels_to_dlabel(precision_map_labels, save_path, label_name=label)


# ----------------------------------------------------------------------------# 
# ----------------           Main Helper Functions            ----------------# 
# ----------------------------------------------------------------------------# 


def create_save_paths(args):
    """ """
    save_paths = {}
    subcortex_status = "_SC" if not args.exclude_subcortex else ""
    mask_tag = "_D{DIST_THRESHOLD}" if args.mask else ""
    tag = f"S{args.sparsity * 10:.0f}{mask_tag}{subcortex_status}"
    prefix = args.prefix
    save_paths["tag"] = tag
    save_paths["label"] = f"{prefix}_{tag}"
    save_paths["FC"] = os.path.join(args.out_dir, f"{prefix}_voxel_FC_{tag}.npz")
    save_paths["parc"] = os.path.join(args.out_dir, f"{prefix}_parcellation_{tag}.npy")
    save_paths["network_assignments"] = os.path.join(args.out_dir, f"{prefix}_{tag}_networks.npy")

    plot_dir = os.path.join(args.out_dir, f"{prefix}_{tag}_plots")
    os.makedirs(plot_dir, exist_ok=True)
    save_paths["QC"] = os.path.join(plot_dir, f"{prefix}_{tag}_QC.png")
    save_paths["surf_plot"] = os.path.join(plot_dir, f"{prefix}_{tag}_surface.png")
    save_paths["dlabel"] = os.path.join(args.out_dir, f"{prefix}_{tag}.dlabel.nii")
    return save_paths


# ----------------------------------------------------------------------------# 
# --------------------                Main                --------------------# 
# ----------------------------------------------------------------------------# 

def check_arguments(args):
    """ """
    # TODO: implement argument checks
    if args.mode in ['generate-corr', "plot", "all"]:
        assert args.ciftis is not None, f"Must provide at least 1 cifti path with -c/--cifti option"
    elif args.mode == 'parcel-detection':
        assert args.corr_matrix is not None, f"Must provide at least 1 correlation matrix with --corr-matrix option"

    if args.mode == "plot":
        assert len(args.partition) > 0, f"Must provide at least 1 partition --partition option for plots"

    return args


def process_arguments(args):
    """ """
    args.out_dir = args.out_dir.strip()
    os.makedirs(args.out_dir.strip(), exist_ok=True)
    args.prefix = args.out_dir.split("/")[-1].strip() if args.prefix is None else args.prefix

    args.exclude_index_path = SUBCORTEX_MASK_PATH if args.exclude_subcortex else None
    args.mask_path = GEODESIC_MASK_GENERIC_PATH.format(dist=DIST_THRESHOLD) if args.mask else None
    return args


def get_arguments(test_args: list = None):
    """
    test_args: easy arg for testing get_arguments function and whole pipeline
    """
    parser = argparse.ArgumentParser(prog='precision-mapping',
                                     description='Generates precision parcellations from a given cifti')

    MODES = ["generate-corr", "parcel-detection", "plot", "all"]

    # General Arguments
    parser.add_argument('-c', "--ciftis", dest='ciftis', action="extend", nargs="+", type=str, required=False,
                            help="Txt file with paths of cifti files or cifti glob path (required for full precision_mapping pipeline)")
    parser.add_argument('-o', "--out", dest='out_dir', action="store", type=str, required=True,
                        help="Output file prefix e.g. 'path/to/dir/file_prefix'")
    parser.add_argument('-p', "--prefix", dest='prefix', action="store", type=str, required=False,
                        help="Output file prefix e.g. 'path/to/dir/file_prefix'")
    parser.add_argument('-v', "--verbose", dest='verbose', action="store", type=int, default=1,
                        required=False, help="Verbosity")
    parser.add_argument('-s', "--seed", dest='seed', action="store", type=int, default=137,
                        required=False, help="Random seed")
    parser.add_argument("--n-reps", dest='n_reps', action="store", type=int, default=1,
                        required=False, help="Number of infomap repetitions")
    parser.add_argument("--sparsity", dest='sparsity', action="store", type=float, default=0.1,
                        required=False, help="FC Sparsity Percent")
    parser.add_argument("--exclude-subcortex", dest='exclude_subcortex', action="store_true", default=False,
                        required=False, help="exclude subcortex")
    parser.add_argument("--mask", dest='mask', action="store_true", default=False,
                        required=False, help="mask")
    parser.add_argument("--dry-run", dest='dry_run', action="store_true", default=False,
                        required=False, help="Runs a dry of the program, checking paths but not doing any anaysis.")
    parser.add_argument("--overwrite", dest='overwrite', action="store_true", default=False,
                        required=False, help="Over writes outputs.")
    #TODO: Identify problems with GLEW library or find way to check (causes seg faults though :/  )
    parser.add_argument("--no-plots", dest='no_plots', action="store_true", default=False,
                        required=False, help="Specifies to skip plotting in case VTK/GLEW lib is messed up.")
    parser.add_argument("-m", '--mode', choices=MODES, type=str, default="all",
                        help=f'Precision mapping options: {MODES}')
    parser.add_argument('--corr-matrix', dest="corr_matrix", action="extend", nargs="+",type=str, required=False,
                            help='Correlation matrix path(s) for parcel-detection')
    parser.add_argument('--partition', dest="partition", action="extend", nargs="+", type=str, required=False,
                            help='Partition path(s) for plots')

    args = parser.parse_args() if test_args is None else parser.parse_args(test_args)
    args = check_arguments(args)
    return process_arguments(args)


def generate_correlation_matrix(args, save_paths):
    """ """
    if args.mode not in ["generate-corr", "all"]:
        return

    args.corr_matrix = [save_paths["FC"]]
    if os.path.exists(save_paths["FC"]) and not args.overwrite:
        print(f"{save_paths['FC']} already exists and no '--overwrite' flag. Skipping correlation matrix creation.")
        return
    
    voxel_data = load_voxel_data(args.ciftis)
    print(f"Loaded ciftis ({len(args.ciftis)})")

    sc = generate_voxel_FC(voxel_data, save_path=save_paths["FC"], sparsity=args.sparsity,
                           exclude_index_path=args.exclude_index_path,
                           mask_path=args.mask_path,
                           block_size=5000)
    print(f"Created Sparse FC matrix ({args.sparsity}%): ", sc.shape)


def parcel_detection(args, save_paths):
    """ """
    if args.mode not in ["parcel-detection", "all"]:
        return
    
    args.partition = save_paths["parc"]
    if os.path.exists(save_paths["parc"]) and not args.overwrite:
        print(f"{save_paths['parc']} already exists and no '--overwrite' flag given. Skipping parcel detection.")
        return

    # TODO: figure out how to make this accepting of mujltiple / if I want accepting of multiple
    sc = scipy.sparse.load_npz(args.corr_matrix[0])
    print(f"Loaded Sparse FC matrix ({args.sparsity}%): ", sc.shape)

    partition = infomap_parcellation(sc, save_path=save_paths["parc"], silent=True,
                                     num_trials=args.n_reps, seed=args.seed)
    
    print(f"Created infomap partition")


def network_assignement(args, save_paths):
    """ """

    if args.mode not in ["all"] or args.no_plots:
        return

    example_cifti = nb.load(args.ciftis[0])
    partition = np.load(args.partition)

    full_vertex_data = load_voxel_data(args.ciftis)
    vertex_data = get_cortex_data(full_vertex_data, example_cifti)
    vertex_labels = get_partition_cortex(partition, example_cifti)

    FC, spatial, network_labels = load_priors()
    vn, vns = get_network_assigments(vertex_labels, vertex_data, network_labels, spatial, FC)
    
    if save_paths["network_assignments"]:
        np.save(save_paths["network_assignments"], [vn, vns])
    
    print("Created network assignments.")


def make_plots(args, save_paths):
    """ """
    if args.mode not in ["plot", "all"] or args.no_plots:
        return

    partition = np.load(args.partition)
    print(f"Loaded infomap partition")
    cifti = nb.load(args.ciftis[0])
    precision_map_values, partition = process_partition(partition, cifti)
    plot_precision_map(precision_map_values, save_path=save_paths["surf_plot"])
    precision_map_QC_plots(partition, save_path=save_paths["QC"])
    # write_dlabel_precision_map(precision_map_values, save_paths["dlabel"], label=save_paths["label"])
    print("Generated precision mapping plots.")


def main(test_args=None):
    """ """
    print()
    args = get_arguments(test_args=test_args)
    save_paths = create_save_paths(args)

    generate_correlation_matrix(args, save_paths)
    parcel_detection(args, save_paths)
    network_assignement(args, save_paths)
    make_plots(args, save_paths)
    print("Done.")


if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
