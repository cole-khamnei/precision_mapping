import argparse
import sys
import os

from src import constants
from src import utils
from src import functional_connectivity as FC
from src import parcellate


sys.path.insert(0, constants.PROJECT_PATH)
import xmath_tools as xmt

# \section constants

FC_FILE_EXT = "FC.npz"
PARCEL_NPY_FILE_EXT = "parcellation.npy"
NETWORK_NPY_FILE_EXT = "networks.npy"

BLOCK_SIZE = 5_000


DEFAULT_SPARSITY = 0.1

DEFAULT_N_INFOMAP_REPS = 1
DEFAULT_N_CORES = 1
DEFAULT_SEED = 137

# \section process arguments


def process_args(args):
    """ """

    args.cifti_paths = utils.resolve_str_txt_list(args.ciftis)
    args.subjects = utils.resolve_str_txt_list(args.subjects_str_txt)

    assert len(args.cifti_paths) == len(args.subjects)

    print(args.cifti_paths)
    print(args.subjects)


    tag_args = dict(sparsity=args.sparsity, mask=args.mask, max_trs=args.max_trs,
                    exclude_subcortex=args.exclude_subcortex)

    args.subject_tags = [utils.create_path_tag(subject, **tag_args) for subject in args.subjects]
    args.subject_out_dirs = [f"{args.out_dir}/{subject}" for subject in args.subjects]
    args.generic_save_paths = [f"{subject_out_dir}/{tag}_{{file_ext}}"
                               for subject_out_dir, tag in zip(args.subject_out_dirs, args.subject_tags)]

    for subject_out_dir in args.subject_out_dirs:
        os.makedirs(subject_out_dir, exist_ok=True)

    args.FC_save_paths = utils.batch_str_format(args.generic_save_paths, file_ext=FC_FILE_EXT)
    args.parcel_save_paths = utils.batch_str_format(args.generic_save_paths, file_ext=PARCEL_NPY_FILE_EXT)
    print(args.FC_save_paths)

    return args


def dry_run_input_check(args):
    """ """

    os.path.exists(args.cifti_paths)



def get_arguments(test_args=None):
    """ """

    # Left these as wide blocks for improved readability :/
    parser = argparse.ArgumentParser(prog='precision-mapping', description='Creates cifti precision maps')
    parser.add_argument('-c', "--ciftis", dest="ciftis", type=str, required=True, help="Txt file with paths of cifti files")
    parser.add_argument('-o', "--out", dest='out_dir', type=str, required=True, help="Output file prefix e.g. 'path/to/dir/file_prefix'")
    parser.add_argument('-s', "--subjects", dest='subjects_str_txt', type=str, help="Output file prefix e.g. 'path/to/dir/file_prefix'")


    parser.add_argument("--sparsity", dest='sparsity', type=float, default=DEFAULT_SPARSITY, help="FC Sparsity Percent")
    parser.add_argument("--exclude-subcortex", dest='exclude_subcortex', action="store_true", default=False, help="exclude subcortex")
    parser.add_argument("--mask", dest='mask', action="store_true", default=False, help="mask")
    parser.add_argument("--max-trs", dest='max_trs', type=int, default=None, help="Maximum TRs to use")


    parser.add_argument("--n-reps", dest='infomap_reps', type=int, default=DEFAULT_N_INFOMAP_REPS, help="Number of infomap repetitions")
    parser.add_argument("--seed", dest='seed', type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--n-cores", dest='n_cores', type=int, default=DEFAULT_N_CORES, help="N cores (used for infomaps section if multiple subjects)")

    parser.add_argument("--overwrite", dest='overwrite', action="store_true", default=False, help="Over writes outputs.")




    args = parser.parse_args() if test_args is None else parser.parse_args(test_args)

    return process_args(args)


# \section main


def main(test_args=None):
    """ """

    args = get_arguments(test_args=test_args)


    FC.generate_correlation_matrix(args.cifti_paths, args.FC_save_paths,
                                   sparsity=args.sparsity,
                                   block_size=BLOCK_SIZE, leave=False,
                                   max_trs=args.max_trs, overwrite=args.overwrite)

    parcellate.parcel_detection(args.FC_save_paths, args.parcel_save_paths,
                                n_cores=args.n_cores, n_reps=args.infomap_reps,
                                overwrite=overwrite, seed=args.seed)



if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
