import numpy as np
import scipy

from tqdm.auto import tqdm


def iterate_over_axis(array, axis=None):
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
    return np.isnan(a).mean(axis=axis).astype(bool) | np.isnan(b).mean(axis=axis).astype(bool)
