import numpy as np
import scipy
import nibabel as nb

from tqdm.auto import tqdm


# ----------------------------------------------------------------------------# 
# --------------------             Np Helpers             --------------------# 
# ----------------------------------------------------------------------------# 


def np_corr(x, y):
    """ """
    x, y = x.T, y.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    y_demeaned = y - y.mean(axis=1, keepdims=True)

    x_norm = x_demeaned / np.sqrt(np.sum(x_demeaned ** 2, axis=1, keepdims=True))
    y_norm = y_demeaned / np.sqrt(np.sum(y_demeaned ** 2, axis=1, keepdims=True))
    return x_norm @ y_norm.T


# ----------------------------------------------------------------------------# 
# --------------------           Cifti Helpers            --------------------# 
# ----------------------------------------------------------------------------# 


def load_voxel_data(dtseries_paths):
    """ """
    if not isinstance(dtseries_paths, str):
        return np.vstack([load_voxel_data(path) for path in dtseries_paths])

    cifti = nb.load(dtseries_paths)
    return cifti.get_fdata()


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
