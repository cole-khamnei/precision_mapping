import argparse

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import functional_connectivity, network_assignment
from src import parcellate, utils
from src import partition_tools as pt

# ----------------------------------------------------------------------------# 
# --------------------              Pipeline              --------------------# 
# ----------------------------------------------------------------------------# 


def full_pipeline(dtseries_paths, subject_ids, sample_labels, out_dir, censor_files=None,
                          overwrite=False, silent=True, block_size=1000, backend="torch",
                          device="cpu", n_cores=1, n_infomaps_reps=50):
    """ """

    dtseries_paths = utils.list_wrap(dtseries_paths, str)
    subject_ids = utils.list_wrap(subject_ids, str)
    sample_labels = utils.list_wrap(sample_labels, str)
    censor_files = None if censor_files is None else utils.list_wrap(sample_labels, str) 

    path_sets = utils.create_pm_paths(subject_ids, sample_labels, out_dir)
    (vertex_fc_paths, parcel_partition_paths, network_partition_paths,
     parcel_dlabel_paths, network_dlabel_paths, plot_save_paths) = path_sets

    functional_connectivity.generate_correlation_matrix(dtseries_paths, vertex_fc_paths, censor_file=censor_files,
                                                        block_size=block_size, overwrite=overwrite,
                                                        backend=backend, device=device)

    parcellate.parcel_detection(vertex_fc_paths, parcel_partition_paths,
                                n_cores=n_cores, n_reps=n_infomaps_reps, 
                                overwrite=overwrite, silent=silent)
    network_assignment.assign_networks_batch(dtseries_paths, parcel_partition_paths,
                                             network_partition_paths,
                                             censor_files=censor_files, 
                                             overwrite=overwrite)

    template_cifti = dtseries_paths[0]

    pt.write_parcel_dlabel(dtseries_paths, parcel_partition_paths, parcel_dlabel_paths, template_cifti)
    pt.write_network_dlabel(dtseries_paths, network_partition_paths, network_dlabel_paths, template_cifti)
    pt.parcel_plot(parcel_partition_paths, network_partition_paths, sample_labels, plot_save_paths, template_cifti)
    return path_sets


# ----------------------------------------------------------------------------# 
# --------------------            Main Helpers            --------------------# 
# ----------------------------------------------------------------------------# 


def process_args(args):
    """ """
    # todo: implement passing .txt?
    return args


def get_arguments(test_args: list = None):
    """
    test_args: easy arg for testing get_arguments function and whole pipeline
    """
    parser = argparse.ArgumentParser(prog='precision-mapping', description='Creates individualized parcellations')

    # General Arguments
    parser.add_argument('-c', "--ciftis", dest='ciftis', action="extend", nargs="+", type=str, required=True,
                            help="Txt file with paths of cifti files or cifti glob path")
    parser.add_argument('-o', "--out", dest='out_dir', action="store", type=str, required=True,
                        help="Output dir e.g. 'path/to/output_dir'")
    parser.add_argument("-i", "--subject-ids", dest='subject_ids', action="extend", nargs="+", type=str,
                        required=True, help="Subject IDs (ex. SUBJECT137) or txt file")
    parser.add_argument("-l", "--sample-labels", dest='sample_labels', action="extend", nargs="+", type=str,
                        required=True, help="Sample labels (SUBJECT137_RUN_1) or txt file")
    
    parser.add_argument("--censor-file", dest='censor_file', action="extend", nargs="+", type=str,
                        required=False, help="Paths to dat files for frame sensoring or txt file")
    
    parser.add_argument("--seed", dest='seed', action="store", type=int, default=137,
                        required=False, help="Random seed")
    parser.add_argument("--n-reps", dest='n_reps', action="store", type=int, default=50,
                        required=False, help="Number of infomap repetitions")
    parser.add_argument("--sparsity", dest='sparsity', action="store", type=float, default=0.1,
                        required=False, help="FC Sparsity Percent")
    parser.add_argument("--overwrite", dest='overwrite', action="store_true", default=False,
                        required=False, help="Over writes outputs.")
    parser.add_argument("--verbose", dest='verbose', action="store_true", default=False,
                        required=False, help="Over writes outputs.")
    parser.add_argument("--no-plots", dest='no_plots', action="store_true", default=False,
                        required=False, help="Specifies to skip plotting in case VTK/GLEW lib is messed up.")


    parser.add_argument("--block-size", dest='block_size', action="store", type=int, default=1000,
                        required=False, help="Block matrix size for acceleration (1000-5000 is reasonable range)")

    parser.add_argument("--device", dest='device', action="store", type=str, required=False, default="cpu",
                        help="torch XLA device to use - options: 'default', 'cpu', 'cuda', 'mps'\nmps == apple silicon gpus")
    parser.add_argument("--backend", dest='backend', type=str, choices=["torch", "numpy"], default="torch",
                        help="block matrix accelleration backend - options: 'torch', 'numpy'")

    parser.add_argument("--n-cores", dest='n_cores', action="store", type=int, required=False, default=1,
                        help="Number of cores to use for infomaps")

    #VTK/GLEW library sometimes causes segfaults on some machines - above my pay grade, so flag to skip plots

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
                  overwrite=args.overwrite, silent=not args.verbose,
                  n_infomaps_reps=args.n_reps,
                  block_size=args.block_size, 
                  device=args.device, backend=args.backend,
                  n_cores=args.n_cores)


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
