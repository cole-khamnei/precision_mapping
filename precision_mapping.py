import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import functional_connectivity, network_assignment
from src import constants, parcellate, plot, utils
from src import partition_tools as pt

# ----------------------------------------------------------------------------# 
# --------------------              Pipeline              --------------------# 
# ----------------------------------------------------------------------------# 


def full_pipeline(dtseries_paths, subject_ids, sample_labels, out_dir,
                  censor_files=None,
                  sparsity=constants.DEFAULT_FC_SPARSITY,
                  overwrite=False, silent=True,
                  block_size=constants.DEFAULT_BLOCK_SIZE,
                  backend="torch", device="cpu",
                  n_cores=constants.DEFAULT_N_CORES,
                  n_infomaps_reps=constants.DEFAULT_N_INFOMAPS_REPS):
    """ """
    dtseries_paths = utils.list_wrap(dtseries_paths, str)
    subject_ids = utils.list_wrap(subject_ids, str)
    sample_labels = utils.list_wrap(sample_labels, str)
    censor_files = None if censor_files is None else utils.list_wrap(censor_files, str)
    paths = utils.create_pm_paths(subject_ids, sample_labels, out_dir)

    functional_connectivity.generate_correlation_matrix(dtseries_paths,
                                                        paths["vertex-fc"],
                                                        censor_file=censor_files,
                                                        sparsity=sparsity,
                                                        block_size=block_size,
                                                        overwrite=overwrite,
                                                        backend=backend,
                                                        device=device)
    parcellate.parcel_detection(paths["vertex-fc"],
                                paths["parcel-partition"],
                                n_cores=n_cores,
                                n_reps=n_infomaps_reps,
                                overwrite=overwrite,
                                silent=silent)
    network_assignment.assign_networks_batch(dtseries_paths,
                                             paths["parcel-partition"],
                                             paths["network-partition"],
                                             censor_files=censor_files, 
                                             overwrite=overwrite)
    template_cifti = dtseries_paths[0]
    pt.write_parcel_dlabel(paths["parcel-partition"],
                           paths["parcel-dlabel"], template_cifti)
    pt.write_network_dlabel(paths["network-partition"],
                            paths["network-dlabel"], template_cifti)
    plot.QC_plots(paths["parcel-partition"], save_path=paths["qc-plot"])
    plot.parcel_plot(paths["parcel-partition"], paths["network-partition"],
                     sample_labels, paths["parcel-plot"], template_cifti)
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

    if args.device == "default":
        args.device = utils.get_available_devices()[0]

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

    #TODO: add seed to pass through to infomaps
    # Infomaps arguments
    parser.add_argument("--seed", dest='seed', action="store", type=int,
                        default=constants.DEFAULT_SEED, help="Random seed")
    parser.add_argument("--n-reps", dest='n_reps', action="store", type=int,
                        default=constants.DEFAULT_N_INFOMAPS_REPS,
                        help="Number of infomap repetitions")
    parser.add_argument("--n-cores", dest='n_cores', action="store", type=int,
                        default=constants.DEFAULT_N_CORES,
                        help="Number of cores to use for infomaps (increases mem usage A LOT)")

    # FC arguments
    parser.add_argument("--sparsity", dest='sparsity', action="store", type=float,
                        default=constants.DEFAULT_FC_SPARSITY,
                        help="FC Sparsity Percent")

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
    print()
    args = get_arguments(test_args=test_args)

    full_pipeline(args.ciftis, args.subject_ids, args.sample_labels, args.out_dir,
                  censor_files=args.censor_files,
                  overwrite=args.overwrite,
                  silent=not args.verbose,
                  sparsity=args.sparsity,
                  n_infomaps_reps=args.n_reps,
                  block_size=args.block_size, 
                  device=args.device,
                  backend=args.backend,
                  n_cores=args.n_cores)


if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
