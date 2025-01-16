import argparse
import os
import sys

import numpy as np
import scipy
import nibabel as nb

from infomap import Infomap

voxel_analysis_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(voxel_analysis_dir, "../")
sys.path.insert(0, project_path)

import torch_math_tools as tmt

#\section constants

DIST_DIR = "/data/data7/network_control/projects/network_control/resources/brain_distances"
SUBCORTEX_MASK_PATH = os.path.join(DIST_DIR, "subcortex_mask.npy")
GEODESIC_MASK_PATH = os.path.join(DIST_DIR, f"geodesic_mask_30.npz")

# \section functions

def load_voxel_data(dtseries_paths):
    """ """
    if not isinstance(dtseries_paths, str):
        return np.vstack([load_voxel_data(path) for path in dtseries_paths])

    cifti = nb.load(dtseries_paths)
    return cifti.get_fdata()


def generate_voxel_FC(voxel_data, save_path=None, sparsity=0.1, exclude_index_path=None, mask_path=None, block_size=5000, **SC_kwargs):
    """ """

    exclude_index = np.load(exclude_index_path) if exclude_index_path else None 
    mask = scipy.sparse.load_npz(mask_path) if mask_path else None

    sc = tmt.matrix.SparseCorrelator.run(voxel_data[:, :], mask=mask, exclude_index=exclude_index,
                                         sparsity_percent=sparsity, block_size=block_size)
    if save_path:
        scipy.sparse.save_npz(save_path, sc)

    return sc


def infomap_parcellation(matrix, save_path=None, num_trials=1, **kwargs):
    """ """

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


def create_save_paths(args):
    """ """
    save_paths = {}
    tag = f"S{args.sparsity * 10:.0f}_D30"
    prefix = args.prefix
    save_paths["tag"] = tag
    save_paths["FC"] = os.path.join(args.out_dir, f"{prefix}_voxel_FC_{tag}.npz")
    save_paths["parc"] = os.path.join(args.out_dir, f"{prefix}_parcellation_{tag}.npy")
    return save_paths


# \section main


def get_arguments(test_args: list = None):
    """
    test_args: easy arg for testing get_arguments function and whole pipeline
    """
    parser = argparse.ArgumentParser(prog='precision-mapping',
                                     description='Generates precision parcellations from a given cifti')
    parser.add_argument('-c', "--ciftis", dest='ciftis', action="extend", nargs="+", type=str, required=True,
                        help="Txt file with paths of cifti files or cifti glob path")
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
    parser.add_argument("--dry-run", dest='dry_run', action="store_true", default=False,
                        required=False, help="Runs a dry of the program, checking paths but not doing any anaysis.")
    parser.add_argument("--overwrite", dest='overwrite', action="store_true", default=False,
                        required=False, help="Over writes outputs.")
    #TODO: Identify problems with GLEW library or find way to check (causes seg faults though :/  )
    parser.add_argument("--no-plots", dest='no_plots', action="store_true", default=False,
                        required=False, help="Specifies to skip plotting in case VTK/GLEW lib is messed up.")

    if test_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(test_args)

    # TODO: implement argument checks
    args.pbar = args.verbose > 0

    args.out_dir = args.out_dir.strip()
    os.makedirs(args.out_dir.strip(), exist_ok=True)

    if args.prefix is None:
        args.prefix = args.out_dir.split("/")[-1].strip()

    return args


def main(test_args=None):
    print()
    args = get_arguments(test_args=test_args)
    save_paths = create_save_paths(args)

    if args.overwrite or not os.path.exists(save_paths["FC"]):
        voxel_data = load_voxel_data(args.ciftis)
        print("Loaded ciftis.")

        sc = generate_voxel_FC(voxel_data, save_path=save_paths["FC"], sparsity=args.sparsity,
                               exclude_index_path=SUBCORTEX_MASK_PATH, mask_path=GEODESIC_MASK_PATH,
                               block_size=5000)
        print(f"Created Sparse FC matrix ({args.sparsity}%): ", sc.shape)
    else:
        sc = scipy.sparse.load_npz(save_paths["FC"])
        print(f"Loaded Sparse FC matrix ({args.sparsity}%): ", sc.shape)

    if args.overwrite or not os.path.exists(save_paths["parc"]):
        partition_arrays = infomap_parcellation(sc, save_path=save_paths["parc"], num_trials=1, seed=args.seed)
        print(f"Created infomap partition_arrays")
    else:
        partition_arrays = np.load(save_paths["parc"])
        print(f"Loaded infomap partition_arrays")

    print("Done.")


if __name__ == '__main__':
    main()

# \section end
