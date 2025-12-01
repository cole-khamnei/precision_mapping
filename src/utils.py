import shutil
import gc
import os

import numpy as np
import multiprocess as mp

from tqdm.auto import tqdm

# ----------------------------------------------------------------------------# 
# --------------------           Random Helpers           --------------------# 
# ----------------------------------------------------------------------------# 


def list_wrap(item, *dtypes):
    """ """
    return [item] if isinstance(item, dtypes) else item


def check_multiple_args(args, main_dtype=str):
    """ """
    if any(not isinstance(arg, main_dtype) for arg in args):
        assert all(len(arg) == len(args[0]) for arg in args[1:]), "arg lists not same length"
        return True

    return False


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


def cache_tmp_path(path, use_cache=True, write_cache=True, cache_dir="~/_tmp", pbar=False, **pb_kwargs):
    """ """
    if not isinstance(path, str):
        iter_ = tqdm(path, desc=f"Caching paths in {cache_dir}", **pb_kwargs) if pbar else path
        return [cache_tmp_path(path_i, use_cache=use_cache, write_cache=write_cache, cache_dir=cache_dir)
                for path_i in iter_]

    if not use_cache:
        return path

    # TODO: builtin hash is non-deterministic, use hashlib or z-adler if want hashed option
    # tmp_path = f"/tmp/cifti_H{hash(path)}.{path.split('.', maxsplit=1)[1]}"

    tmp_path = f"{cache_dir}/" + path.split("/data/data", maxsplit=1)[1].lstrip("1234567/").replace("/", "--")
    tmp_path = os.path.expanduser(tmp_path)

    if not os.path.exists(tmp_path):
        if write_cache:
            shutil.copyfile(path, tmp_path)
            gc.collect()
        else:
            tmp_path = path

    return tmp_path


def read_txt(txt_path: str) -> list:
    """ """

    with open(txt_path, "r") as file:
        return file.read().strip().split()


def resolve_str_txt_list(str_txt_list, file_ext=None):
    """ """
    assert isinstance(str_txt_list, str)

    if str_txt_list.endswith(".txt"):
        list_items = read_txt(str_txt_list)

    else:
        list_items = [str_txt_list]

    if file_ext:
        assert all(item.endswith(file_ext) for item in list_items)

    return list_items


def batch_str_format(str_list, **kwargs):
    """ """
    return [str_i.format(**kwargs) for str_i in str_list]


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
# -----------------           Multiprocess Helpers           -----------------# 
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


def create_path_tag(prefix, sparsity, mask, exclude_subcortex, max_trs=None, dist_threshold=10):
    """ """
    
    subcortex_status = "_SC" if not exclude_subcortex else ""
    mask_tag = f"_D{dist_threshold}" if mask else ""
    max_trs_tag = f"_TR{max_trs:0.0f}" if max_trs else ""
    tag = f"S{sparsity * 10:.0f}{mask_tag}{subcortex_status}{max_trs_tag}"
    
    return f"{prefix}_{tag}"


def create_pm_paths(subject_ids, sample_labels, precision_maps_out_dir):
    """ """

    assert_exists(precision_maps_out_dir)
    subject_ids = list_wrap(subject_ids, str)
    sample_labels = list_wrap(sample_labels, str)

    vertex_fc_paths, parcel_partition_paths, network_partition_paths = [], [], []
    parcel_dlabel_paths, network_dlabel_paths, plot_save_paths = [], [], []

    for subject_id, sample_label in zip(subject_ids, sample_labels):
        subject_pm_dir = os.path.join(precision_maps_out_dir, subject_id)
        if not os.path.exists(subject_pm_dir):
            os.mkdir(subject_pm_dir)

        subject_generic_file_name = f"{subject_pm_dir}/{sample_label}_{{file_ending}}"
        
        vertex_fc_paths.append(subject_generic_file_name.format(file_ending="vertex_FC.npz"))
        parcel_partition_paths.append(subject_generic_file_name.format(file_ending="parcel_partition.npy"))
        network_partition_paths.append(subject_generic_file_name.format(file_ending="network_partition.npy"))
        parcel_dlabel_paths.append(subject_generic_file_name.format(file_ending="parcels.dlabel.nii"))

        network_dlabel_paths.append(subject_generic_file_name.format(file_ending="networks.dlabel.nii"))
        plot_save_paths.append(subject_generic_file_name.format(file_ending="parcellation_plot.png"))

    return (vertex_fc_paths, parcel_partition_paths, network_partition_paths,
            parcel_dlabel_paths, network_dlabel_paths, plot_save_paths)


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
