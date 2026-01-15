import os

import numpy as np
import multiprocess as mp

import termcolor

from tqdm.auto import tqdm
# from termcolor import colored

from . import constants

# ----------------------------------------------------------------------------# 
# ------------             Multiple Argument Helpers              ------------# 
# ----------------------------------------------------------------------------# 


def list_wrap(item, *dtypes) -> list:
    """ """
    return [item] if isinstance(item, dtypes) else item


def check_multiple_args(args, main_dtype=str):
    """ """
    if any(not isinstance(arg, main_dtype) for arg in args):
        assert all(len(arg) == len(args[0]) for arg in args[1:]), "arg lists not same length"
        return True

    return False


def multicall(func, *args, main_dtype=str, pbar=False, pbar_kwargs={}, **kwargs):
    """ """
    if check_multiple_args(args, main_dtype=main_dtype):

        if pbar:

            if "colour" not in pbar_kwargs:
                pbar_kwargs["colour"] = constants.MAIN_PBAR_COLOR

            if "desc" in pbar_kwargs:
                pbar_kwargs["desc"] = colored(pbar_kwargs["desc"], pbar_kwargs["colour"])

            for args_i in tqdm(zip(*args), total=len(args[0]), **pbar_kwargs):
                func(*args_i, **kwargs)
        else:
            np.vectorize(func)(*args, **kwargs)
        return True

    return False


# ----------------------------------------------------------------------------# 
# --------------            Input Txt File Handlers             --------------# 
# ----------------------------------------------------------------------------# 


def read_txt(txt_path: str) -> list:
    """ """
    with open(txt_path, "r") as file:
        return file.read().strip().split()


def resolve_str_txt_list(item_list, file_ext=None):
    """ """
    if item_list is None:
        return None

    if item_list[0].endswith(".txt"):
        assert len(item_list) == 1, f"only one txt file per argument set\n{item_list}"
        item_list = read_txt(item_list[0])

    if file_ext:
        assert all(item.endswith(file_ext) for item in item_list)

    return item_list


# ----------------------------------------------------------------------------# 
# --------------------              Printer               --------------------# 
# ----------------------------------------------------------------------------# 


class Printer:
    def __init__(self, silent=False):
        self.silent = silent

    def __call__(self, *args):
        """ """
        if not self.silent:
            print(*args)

    def mute(self):
        self.silent = True

    def unmute(self):
        self.silent = False


printer = Printer(silent=False)


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def colored(text, color):
    """ """
    if not is_interactive():
        return termcolor.colored(text, color)

    return text


# ----------------------------------------------------------------------------# 
# --------------------             Np Helpers             --------------------# 
# ----------------------------------------------------------------------------# 


def np_corr(x, y):
    """ """
    x, y = x.T, y.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    y_demeaned = y - y.mean(axis=1, keepdims=True)

    sigma_x = np.sqrt(np.sum(x_demeaned ** 2, axis=1, keepdims=True))
    sigma_y = np.sqrt(np.sum(y_demeaned ** 2, axis=1, keepdims=True))

    sigma_x = (sigma_x == 0) * 1 + sigma_x
    sigma_y = (sigma_y == 0) * 1 + sigma_y
    assert np.all(sigma_x != 0)
    assert np.all(sigma_y != 0)

    x_norm = x_demeaned / sigma_x
    y_norm = y_demeaned / sigma_y
    return x_norm @ y_norm.T


# ----------------------------------------------------------------------------# 
# ------------             Device/multiprocess Tools              ------------# 
# ----------------------------------------------------------------------------# 


def get_n_cores(n_cores=None, cpu_offset=1):
    """ """
    max_cpus = mp.cpu_count()
    if n_cores is None:
        return max_cpus - cpu_offset

    return min(max_cpus, n_cores)


# ----------------------------------------------------------------------------# 
# --------------------            Path Helpers            --------------------# 
# ----------------------------------------------------------------------------# 


def assert_exists(path):
    """ """
    assert os.path.exists(path), f"'{path}' does not exist."


def create_pm_paths(subject_ids, sample_labels, precision_maps_out_dir, method="infomaps"):
    """ """
    assert_exists(precision_maps_out_dir)
    subject_ids = list_wrap(subject_ids, str)
    sample_labels = list_wrap(sample_labels, str)

    pm_tag = "_kms" if method == "kmeans" else ""

    path_sets = {output: [] for output in constants.OUTPUT_FILE_ENDINGS.keys()}
    for subject_id, sample_label in zip(subject_ids, sample_labels):
        subject_pm_dir = os.path.join(precision_maps_out_dir, subject_id)
        if not os.path.exists(subject_pm_dir):
            os.mkdir(subject_pm_dir)

        for output, file_ending in constants.OUTPUT_FILE_ENDINGS.items():
            path_sets[output].append(f"{subject_pm_dir}/{sample_label}{pm_tag}_{file_ending}")

    return path_sets


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
