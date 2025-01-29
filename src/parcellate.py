import numpy as np
import multiprocess as mp
# import multiprocessing as mp

from infomap import Infomap

from . import utils

# ----------------------------------------------------------------------------# 
# ----------------           Infomaps Parcellating            ----------------# 
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
        # print(f"infomap {save_path} done.")
    
    return index, values


def batch_infomap_parcellation(matrices, save_paths, n_cores=None, **infomap_kwargs):
    """ """
    n_cores = utils.get_n_cores(n_cores)

    arg_sets = zip(matrices, save_paths)

    single_parcel_func = lambda args: infomap_parcellation(args[0], args[1], silent=True, **infomap_kwargs)
    # single_parcel_func = lambda args: args[1]

    with mp.Pool(n_cores) as p:
        results = p.map_async(single_parcel_func, arg_sets)
        # results = p.map(single_parcel_func, save_paths)
        # results = p.map_async(single_parcel_func, save_paths)
        # results = p.map_async(lambda s: s ** 2, np.arange(10))
        # results = results.get(timeout=1)
        results = results.get()
    # print(list(results))
    return results
    raise NotImplementedError


# ----------------------------------------------------------------------------# 
# --------------------               Tests                --------------------# 
# ----------------------------------------------------------------------------# 


def main():
    """ tests """
    mp.cpu_count()

    print(utils.get_n_cores())



if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
