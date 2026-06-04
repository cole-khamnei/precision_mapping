import potpourri3d as pp3d
import nibabel as nb
import numpy as np
import multiprocess as mp

from scipy.sparse import lil_matrix
from tqdm.auto import tqdm


def build_sparse_geodesic_matrix(coords, triangles, max_dist=50.0, idx_range=None):
    """ """
    solver = pp3d.MeshHeatMethodDistanceSolver(coords, triangles)
    n = len(coords)
    D = lil_matrix((n, n), dtype=np.float32)

    idx_range = range(n) if idx_range is None else idx_range
    for i in tqdm(idx_range):
        dists = solver.compute_distance(i)
        mask = (dists > 0) & (dists <= max_dist)
        D[i, mask] = dists[mask].astype(np.float32)
    return D.tocsr()


def parallel_build_sparse_geodesic_matrix(coords, triangles, max_dist=50.0, n_cpu=8, n_splits=100):
    """ """
    compute_chunk = lambda idx_range: build_sparse_geodesic_matrix(coords, triangles, max_dist=max_dist, idx_range=idx_range)

    idx_range_sets = np.array_split(np.arange(len(coords)), n_splits)
    with mp.Pool(processes=n_cpu) as pool:
        D = sum(tqdm(pool.imap(compute_chunk, idx_range_sets), total=len(idx_range_sets)))

    return D