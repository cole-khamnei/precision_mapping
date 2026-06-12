import potpourri3d as pp3d
import nibabel as nb
import numpy as np
import multiprocess as mp

import scipy
from scipy.sparse import lil_matrix
from tqdm.auto import tqdm

from . import constants
from .utils import colored

def build_sparse_geodesic_matrix(coords, triangles, max_dist=50.0, idx_range=None):
    """ """
    solver = pp3d.MeshHeatMethodDistanceSolver(coords, triangles)
    n = len(coords)
    D = lil_matrix((n, n), dtype=np.float32)

    idx_range = range(n) if idx_range is None else idx_range
    for i in idx_range:
        dists = solver.compute_distance(i)
        mask = (dists > 0) & (dists <= max_dist)
        D[i, mask] = dists[mask].astype(np.float32)
    return D.tocsr()


def parallel_build_sparse_geodesic_matrix(surface_path, max_dist=20.0, n_cpu=8, n_splits=100,
                                          pbar=True, hl="", pbar_kwargs=dict(leave=False)):
    """ """
    surface = nb.load(surface_path)
    coords = surface.darrays[0].data.astype(np.float64)
    triangles = surface.darrays[1].data.astype(np.int32)

    compute_chunk = lambda idx_range: build_sparse_geodesic_matrix(coords, triangles, max_dist=max_dist, idx_range=idx_range)

    idx_range_sets = np.array_split(np.arange(len(coords)), n_splits)
    # idx_range_sets = np.array_split(np.arange(1), n_splits)
    with mp.Pool(processes=n_cpu) as pool:
        if pbar:
            pbar_kwargs["desc"] = colored(f"Calculating {hl} hemisphere distances", "cyan")
            pbar_kwargs["colour"] = "#008080"
            D = sum(tqdm(pool.imap(compute_chunk, idx_range_sets), total=len(idx_range_sets), **pbar_kwargs))
        else:
            D = sum(pool.imap(compute_chunk, idx_range_sets))

    return D


def calc_surf_distance_matrix(left_surface_paths, right_surface_paths, dist_matrix_paths, distance_threshold=10,
                              method="approximate"):
    """ """
    assert method in ["approximate", "exact"]

    if not isinstance(left_surface_paths, str):
        assert len(left_surface_paths) == len(right_surface_paths)
        assert len(right_surface_paths) == len(dist_matrix_paths)

        pbar = tqdm(total=len(dist_matrix_paths),
                desc=colored("Building distance matrices", constants.MAIN_TERM_COLOR),
                colour=constants.MAIN_PBAR_COLOR)

        res = []
        for args in zip(left_surface_paths, right_surface_paths, dist_matrix_paths):
            res.append(calc_surf_distance_matrix(*args, method="approximate"))
            pbar.update(1)
        pbar.close()

        return res

    if method == "approximate":
        # approximate method is has <0.1% error if using a distance threshold of 10 mm, so effect on pfm is that or smaller

        build_geodesic_matrix_kwargs = dict(max_dist=50.0, n_cpu=8, n_splits=100)
        D_left = parallel_build_sparse_geodesic_matrix(left_surface_paths, hl="left", **build_geodesic_matrix_kwargs)
        D_right = parallel_build_sparse_geodesic_matrix(right_surface_paths, hl="right", **build_geodesic_matrix_kwargs)

        D_cortex = scipy.sparse.block_diag([D_left, D_right])

        # following is only true if fully ignoring subcortex
        # TODO: fully ignore subcortex?
        D_full = scipy.sparse.coo_matrix((D_cortex.data, (D_cortex.row, D_cortex.col)), shape=(91_282, 91_282)).tocsr()
        scipy.sparse.save_npz(dist_matrix_paths, D_full)

        mask_npz = D_full.copy()
        csr_mask = mask_npz.data < distance_threshold
        mask_npz.data[csr_mask] = 0
        mask_npz.eliminate_zeros()
        dist_mask_path = dist_matrix_paths.replace(".npz", f"_mask_{distance_threshold}.npz")
        scipy.sparse.save_npz(dist_mask_path, mask_npz)

        return dist_mask_path

    else:
        raise NotImplementedError
