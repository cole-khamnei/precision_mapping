import numpy as np

from infomap import Infomap

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
    
    return index, values


def batch_infomap_parcellation(matrices, save_paths, n_cores=None, **infomap_kwargs):
    """ """
    raise NotImplementedError


# ----------------------------------------------------------------------------# 
# --------------------               Tests                --------------------# 
# ----------------------------------------------------------------------------# 


def main():
    """ tests """


if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
