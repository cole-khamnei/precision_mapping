import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import functional_connectivity, network_assignment
from src import constants, parcellate, plot, utils
from src import partition_tools as pt
from src import spatial_filtering, distances

from src.utils import colored

# ----------------------------------------------------------------------------# 
# --------------------              Pipeline              --------------------# 
# ----------------------------------------------------------------------------# 


def infomaps_parcel_detection(dtseries_paths,
                              vertex_fc_paths,
                              parcel_detection_save_paths,
                              paths,
                              censor_files, sparsity,
                              mask, block_size, overwrite, backend, device,
                              n_cores, n_infomaps_reps, silent, seed, **kwargs):
    """ """
    functional_connectivity.generate_correlation_matrix(dtseries_paths,
                                                        # paths["vertex-fc"],
                                                        vertex_fc_paths,
                                                        censor_file=censor_files,
                                                        sparsity=sparsity,
                                                        mask=mask,
                                                        block_size=block_size,
                                                        overwrite=overwrite,
                                                        backend=backend,
                                                        device=device)
    parcellate.parcel_detection(
                                # paths["vertex-fc"],
                                # paths["parcel-partition"],
                                vertex_fc_paths,
                                parcel_detection_save_paths,
                                n_cores=n_cores,
                                n_reps=n_infomaps_reps,
                                overwrite=overwrite,
                                silent=silent,
                                seed=seed)


def full_pipeline(dtseries_paths, subject_ids, sample_labels, out_dir,
                  censor_files=None,
                  left_surface_paths=None,
                  right_surface_paths=None,
                  overwrite=False,
                  silent=True,
                  method="infomaps",
                  sparsity=constants.DEFAULT_FC_SPARSITY,
                  mask=constants.DEFAULT_MASK,
                  backend="torch",
                  device="default",
                  block_size=constants.DEFAULT_BLOCK_SIZE,
                  seed=constants.DEFAULT_SEED,
                  n_cores=constants.DEFAULT_N_CORES,
                  n_cores_wb=constants.DEFAULT_N_CORES_WB,
                  n_infomaps_reps=constants.DEFAULT_N_INFOMAPS_REPS,
                  n_parcels=constants.DEFAULT_K_PARCELS,
                  spatial_filter_size=constants.SPATIAL_FILTER_SIZE,
                  n_spatial_filter_parcels=None):
    """ """
    dtseries_paths = utils.list_wrap(dtseries_paths, str)
    subject_ids = utils.list_wrap(subject_ids, str)
    sample_labels = utils.list_wrap(sample_labels, str)
    censor_files = None if censor_files is None else utils.list_wrap(censor_files, str)

    left_surface_paths = utils.list_wrap(left_surface_paths, str)
    right_surface_paths = utils.list_wrap(right_surface_paths, str)
    
    paths = utils.create_pm_paths(subject_ids, sample_labels, out_dir, method=method)

    if silent:
        utils.printer.mute()

    #TODO: add include index mapping
    #TODO: wrap this in parcellate func, which specifies parcellate method (infomaps, kmeans, etc)

    #TODO: clean up following spatial filter logic flow
    paths["parcel-partition-no-sfd"] = [p.replace(".npy", "_no_sfd.npy") for p in paths["parcel-partition"]]
    if spatial_filter_size > 0:
        parcel_detection_save_paths = paths["parcel-partition-no-sfd"]
    else:
        parcel_detection_save_paths = paths["parcel-partition"]

    paths["distance-mask"] = distances.calc_surf_distance_matrix(left_surface_paths, right_surface_paths, 
                                                                 paths["distance-matrix"], distance_threshold=10)

    if method == "infomaps":
        infomaps_parcel_detection(dtseries_paths,
                                  paths["vertex-fc"],
                                  parcel_detection_save_paths,
                                  paths,
                                  censor_files, sparsity, paths["distance-mask"], block_size,
                                  overwrite, backend, device, n_cores, n_infomaps_reps, silent, seed)
    elif method == "kmeans":
        parcellate.kmeans_parcel_detection(dtseries_paths, paths["parcel-partition"],
                                           censor_files, overwrite, seed, n_parcels)
    else:
        raise NotImplementedError(f"invalid method: '{method}'")


    if spatial_filter_size > 0:
        spatial_filtering.batch_spatial_filter_and_dilate(parcel_detection_save_paths, paths["parcel-partition"], overwrite=overwrite, 
                                                          n_cpu=n_cores_wb, pbar=True, min_size=spatial_filter_size,
                                                          n_parcels=n_spatial_filter_parcels)


    network_assignment.assign_networks_batch(dtseries_paths,
                                             paths["parcel-partition"],
                                             paths["network-partition"],
                                             censor_files=censor_files, 
                                             overwrite=overwrite)

    o_kwargs = dict(template_cifti=dtseries_paths[0], overwrite=overwrite)
    pt.write_parcel_dlabel(paths["parcel-partition"], paths["parcel-dlabel"], **o_kwargs)
    pt.write_network_dlabel(paths["network-partition"], paths["network-dlabel"], **o_kwargs)
    pt.calculate_network_surface_areas(paths["network-partition"], paths["network-size-csv"], left_surface_paths, right_surface_paths, **o_kwargs)
    plot.QC_plots(paths["parcel-partition"], save_path=paths["qc-plot"], overwrite=overwrite)
    plot.parcel_plot(paths["parcel-partition"], paths["network-partition"],
                     sample_labels, paths["parcel-plot"], **o_kwargs)
    return paths


# ----------------------------------------------------------------------------# 
# --------------------            Main Helpers            --------------------# 
# ----------------------------------------------------------------------------# 


def process_args(args):
    """ """
    args.ciftis = utils.resolve_str_txt_list(args.ciftis, file_ext=".dtseries.nii")
    args.subject_ids = utils.resolve_str_txt_list(args.subject_ids)
    args.sample_labels = utils.resolve_str_txt_list(args.sample_labels)
    args.censor_files = utils.resolve_str_txt_list(args.censor_files, file_ext=".dat")
    args.left_surface_paths = utils.resolve_str_txt_list(args.left_surface_paths, file_ext=".surf.gii")
    args.right_surface_paths = utils.resolve_str_txt_list(args.right_surface_paths, file_ext=".surf.gii")

    if args.mask.strip().lower() == "none":
        args.mask = None

    return args


def get_arguments(test_args: list = None):
    """
    test_args: easy arg for testing get_arguments function and whole pipeline
    """
    parser = argparse.ArgumentParser(prog='precision-mapping', description='Creates individualized parcellations')

    # General Arguments
    parser.add_argument('-c', "--ciftis", dest='ciftis', action="extend", nargs="+", required=True,
                        help="Txt file with paths of cifti files or cifti glob path")
    parser.add_argument('-o', "--out", dest='out_dir', action="store", required=True,
                        help="Output dir e.g. 'path/to/output_dir'")
    parser.add_argument("-i", "--subject-ids", dest='subject_ids', action="extend", nargs="+",
                        required=True, help="Subject IDs (ex. SUBJECT137) or txt file")
    parser.add_argument("-l", "--sample-labels", dest='sample_labels', action="extend", nargs="+",
                        required=True, help="Sample labels (SUBJECT137_RUN_1) or txt file")
    parser.add_argument("--censor-files", dest='censor_files', action="extend", nargs="+",
                        help="Paths to dat files for frame sensoring or txt file (inclusion list - with 1s being include and in a row or column tsv form.)")
    parser.add_argument("--left-surface-files", dest='left_surface_paths', required=True, action="extend", nargs="+",
                        help="Paths to left surface files")
    parser.add_argument("--right-surface-files", dest='right_surface_paths', required=True, action="extend", nargs="+",
                        help="Paths to left surface files")

    # Cluster method:
    parser.add_argument("--method", dest='method', choices=["infomaps", "kmeans"], default="infomaps",
                        help="Clustering method options: 'infomaps', 'kmeans'")
    parser.add_argument("--k-parcels", "-k", dest='k_parcels', action="store", type=int,
                        default=constants.DEFAULT_K_PARCELS, help="K number of parcels (for kmeans method)")

    #TODO: add seed to pass through to infomaps/other clustering algorithm
    # Infomaps arguments
    parser.add_argument("--seed", dest='seed', action="store", type=int,
                        default=constants.DEFAULT_SEED, help="Random seed")
    parser.add_argument("--n-reps", dest='n_reps', action="store", type=int,
                        default=constants.DEFAULT_N_INFOMAPS_REPS,
                        help="Number of infomap repetitions")
    parser.add_argument("--n-cores", dest='n_cores', action="store", type=int,
                        default=constants.DEFAULT_N_CORES,
                        help="Number of cores to use for infomaps (increases mem usage A LOT!)")

    parser.add_argument("--spatial-filter-size", dest='spatial_filter_size', action="store", type=float,
                        default=constants.SPATIAL_FILTER_SIZE,
                        help="Spatial filter size (mm)")
    parser.add_argument("--spatial-filter-n-parcels", dest='n_spatial_filter_parcels', action="store", type=int,
                        default=None,
                        help="Spatial filter n parcels to filter for debugging (default is all, just slow)")

    # FC arguments
    parser.add_argument("--sparsity", dest='sparsity', action="store", type=float,
                        default=constants.DEFAULT_FC_SPARSITY, help="FC Sparsity Percent")

    parser.add_argument("--mask", dest='mask', action="store", type=str,
                        default=constants.DEFAULT_MASK,
                        help="FC vertex  distance mask (sparse matrix (npz) that"
                             "specifies vertex pairs to exclude (1)")

    # FC acceleration args
    parser.add_argument("--block-size", dest='block_size', action="store", type=int,
                        default=constants.DEFAULT_BLOCK_SIZE,
                        help="Block matrix size for acceleration (1000-5000 is reasonable range)")
    parser.add_argument("--device", dest='device', action="store", default="default",
                        help="torch device - options: 'default', 'cpu', 'cuda', 'mps'"
                        "\nmps == apple silicon gpus")
    parser.add_argument("--backend", dest='backend', choices=["torch", "numpy"], default="torch",
                        help="block matrix accelleration backend - options: 'torch', 'numpy'")

    # pipeline arguments
    parser.add_argument("--overwrite", dest='overwrite', action="store_true",
                        help="Over writes outputs.")
    parser.add_argument("--verbose", dest='verbose', action="store_true",
                        help="Over writes outputs.")
    #VTK/GLEW library sometimes causes segfaults on some machines - above my pay grade, so flag to skip plots
    parser.add_argument("--no-plots", dest='no_plots', action="store_true",
                        help="Specifies to skip plotting in case VTK/GLEW lib is messed up.")



    args = parser.parse_args() if test_args is None else parser.parse_args(test_args)
    args = process_args(args)
    return args


# ----------------------------------------------------------------------------# 
# --------------------                Main                --------------------# 
# ----------------------------------------------------------------------------# 


def main(test_args=None):
    """ """
    print(colored("\nRunning PFM pipeline:", "yellow"))
    args = get_arguments(test_args=test_args)

    full_pipeline(args.ciftis, args.subject_ids, args.sample_labels, args.out_dir,
                  censor_files=args.censor_files,
                  left_surface_paths=args.left_surface_paths,
                  right_surface_paths=args.right_surface_paths,
                  overwrite=args.overwrite,
                  silent=not args.verbose,
                  method=args.method,
                  sparsity=args.sparsity,
                  mask=args.mask,
                  n_infomaps_reps=args.n_reps,
                  seed=args.seed,
                  block_size=args.block_size, 
                  device=args.device,
                  backend=args.backend,
                  n_cores=args.n_cores,
                  n_parcels=args.k_parcels,
                  spatial_filter_size=args.spatial_filter_size,
                  n_spatial_filter_parcels=args.n_spatial_filter_parcels)
    print(colored("PFM pipeline finished.\n", "yellow"))

if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
