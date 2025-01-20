import numpy as np
import scipy
import nibabel as nb

from tqdm.auto import tqdm


def iterate_over_axis(array, axis=None):
    """ """
    return (np.take(array, i, axis=axis) for i in tqdm(range(array.shape[axis]), leave=False))


def nan_pearsonr(a, b, axis=None):
    """ """
    assert a.shape == b.shape
    if a.ndim > 1 and axis is None:
        return nan_pearsonr(a.ravel(), b.ravel())
    elif a.ndim > 1:
        print("dim 1")
        axis, new_axis = (axis, None) if isinstance(axis, int) else (axis[0], axis[1])
        a_iter = iterate_over_axis(a, axis=axis)
        b_iter = iterate_over_axis(b, axis=axis)
        return np.array([nan_pearsonr(a, b, axis=new_axis) for a, b in zip(a_iter, b_iter)])
    else:
        ni = ~ (np.isnan(a) | np.isnan(b))
        print(np.where(~ni))
        assert False
        return scipy.stats.pearsonr(a[ni], b[ni], axis=0)[0]

# nan_pearsonr(vf_values, voxelf_values, axis=0)


def get_shared_nan_index(a, b, axis=0):
    """ """
    return np.isnan(a).mean(axis=axis).astype(bool) | np.isnan(b).mean(axis=axis).astype(bool)


# \section np helpers


def np_corr(x, y):
    """ """
    x, y = x.T, y.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    y_demeaned = y - y.mean(axis=1, keepdims=True)

    x_norm = x_demeaned / np.sqrt(np.sum(x_demeaned ** 2, axis=1, keepdims=True))
    y_norm = y_demeaned / np.sqrt(np.sum(y_demeaned ** 2, axis=1, keepdims=True))
    return x_norm @ y_norm.T


# \section cifti helpers

def load_voxel_data(dtseries_paths):
    """ """
    if not isinstance(dtseries_paths, str):
        return np.vstack([load_voxel_data(path) for path in dtseries_paths])

    cifti = nb.load(dtseries_paths)
    return cifti.get_fdata()

# \section end
