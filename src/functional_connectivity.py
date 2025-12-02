import os

import numpy as np
import scipy

from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from . import constants, cifti_tools, utils
from .sparse_correlator import SparseCorrelator


# ----------------------------------------------------------------------------# 
# --------               Functional Connectivity Tools                --------# 
# ----------------------------------------------------------------------------# 


def generate_voxel_FC(voxel_data,
                      save_path=None,
                      sparsity=0.1,
                      exclude_index_path=None,
                      mask=None,
                      block_size=5_000,
                      leave=True,
                      **SC_kwargs):
    """ """
    exclude_index = np.load(exclude_index_path) if exclude_index_path else None

    if str(mask).isdigit():
        mask = constants.get_geodesic_distance_mask(int(mask))

    mask = scipy.sparse.load_npz(mask) if mask else None

    # TODO: Fix masking related issues
    # TODO: add infomaps check, to insure that at least a certain percent of vertices have connections

    sc = SparseCorrelator.run(voxel_data, sparsity_percent=sparsity, symmetric=True,
                              mask=mask, exclude_index=exclude_index,
                              block_size=block_size, leave=leave, **SC_kwargs)
    if save_path:
        scipy.sparse.save_npz(save_path, sc)

    return sc


def generate_correlation_matrix(cifti_path, save_path,
                                censor_file=None,
                                sparsity=0.1,
                                max_trs=None,
                                exclude_index_path=None,
                                mask=None,
                                block_size=5_000,
                                overwrite=False,
                                leave=False,
                                **SC_kwargs):
    """ """
    if isinstance(cifti_path, str) and isinstance(save_path, str):
        pass
    elif isinstance(cifti_path, str) or isinstance(save_path, str):
        raise ValueError(f"Both cifti_path: {cifti_path} and save_path: {save_path} must be same type")
    else:
        assert len(cifti_path) == len(save_path), f"Path variables have diff lens: {len(cifti_path)}, {len(save_path)}"

        return generate_correlation_batch(cifti_path, save_path, censor_files=censor_file,
                                           sparsity=sparsity, max_trs=max_trs,
                                           exclude_index_path=exclude_index_path, mask=mask,
                                           block_size=block_size, overwrite=overwrite,
                                           leave=leave, **SC_kwargs)


    if os.path.exists(save_path) and not overwrite:
        utils.printer(f"{save_path} already exists and no '--overwrite' flag. Skipping correlation matrix creation.")
        return save_path
    
    voxel_data = cifti_tools.load_voxel_data(cifti_path, censor_file=censor_file)

    if max_trs:
        voxel_data = voxel_data[:max_trs]

    generate_voxel_FC(voxel_data, save_path=save_path, sparsity=sparsity,
                      exclude_index_path=exclude_index_path,
                      mask=mask, block_size=block_size,
                      leave=leave, **SC_kwargs)

    return save_path


def generate_correlation_batch(cifti_paths, save_paths,
                               censor_files=None,
                               sparsity=0.1,
                               max_trs=None,
                               exclude_index_path=None,
                               mask=None,
                               block_size=5_000,
                               overwrite=False,
                               leave=False,
                               **SC_kwargs):
    """ too optimized lol, simpler and slower is often better -.-"""
    assert all(os.path.exists(os.path.dirname(path)) for path in save_paths)

    results = []
    pbar = tqdm(total=len(save_paths), desc="Generating vertex-level FC")

    load_indices = [i for i, save_path in enumerate(save_paths) if not os.path.exists(save_path) or overwrite]
    written_save_paths = [save_path for save_path in save_paths if os.path.exists(save_path) and not overwrite]

    pbar.update(len(written_save_paths))
    if len(written_save_paths) == len(save_paths):
        pbar.close()
        return written_save_paths

    censor_files = [None] * len(cifti_paths) if censor_files is None else censor_files

    with ThreadPoolExecutor(max_workers=3) as executor:

        future_voxel_data = executor.submit(cifti_tools.load_voxel_data,
                                            cifti_paths[load_indices[0]],
                                            censor_files[load_indices[0]])
        for j, load_index in enumerate(load_indices):
            save_path = save_paths[load_index]
            voxel_data = future_voxel_data.result()
            if j < len(load_indices) - 1:
                future_voxel_data = executor.submit(cifti_tools.load_voxel_data,
                                                    cifti_paths[load_indices[j + 1]],
                                                    censor_files[load_indices[0]])

            if os.path.exists(save_path) and not overwrite:
                utils.printer(f"{save_path} already exists and no '--overwrite' flag. Skipping correlation matrix creation.")
                results.append(save_path)
                continue

            voxel_data = voxel_data[:max_trs] if max_trs else voxel_data
            sc = generate_voxel_FC(voxel_data, save_path=None, sparsity=sparsity,
                                   exclude_index_path=exclude_index_path,
                                   mask=mask,
                                   block_size=block_size, leave=leave, **SC_kwargs)

            # Asynchronously save files (takes ~3 seconds on 6Gb/s NAS)
            executor.submit(scipy.sparse.save_npz, save_path, sc)

            results.append(save_path)
            pbar.update(1)

        executor.shutdown(wait=True)
        pbar.close()

    return results


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
