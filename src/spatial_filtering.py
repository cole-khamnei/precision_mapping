import os
import subprocess
import time

import numpy as np
import nibabel as nb
import multiprocess as mp

from tqdm.auto import tqdm


from . import constants, partition_tools, parcellate
from . import surface_mapping as sfm
from .utils import colored


def remove_files(files):
    """ """
    for file in files:
        if os.path.exists(file):
            os.remove(file)



def write_dtseries_cifti(values: np.ndarray, save_path: str, template_cifti, tr: float = 1.0):
    """ """
    assert save_path.endswith(".dtseries.nii")
    assert template_cifti is not None, "Must provide 'template_cifti'"

    values = np.expand_dims(values, 0) if values.ndim == 1 else values
    assert values.shape[1] in [91282, 59412], f"Can only take n x 91282 or n x 59412 sized arrays (given {values.shape[1]})"

    if values.shape[1] == 59412:
        values = np.hstack([values, np.zeros((len(values), 91282 - 59412))])

    if isinstance(template_cifti, str):
        template_cifti = nb.load(template_cifti)

    brain_axis = template_cifti.header.get_axis(1)
    series_axis = nb.cifti2.SeriesAxis(start=0.0, step=tr, size=values.shape[0], unit="SECOND")
    header = nb.cifti2.Cifti2Header.from_axes((series_axis, brain_axis))
    dtseries = nb.Cifti2Image(values, header=header)
    dtseries.to_filename(save_path)


def wb_spatial_filter(values, template_cifti, min_size=50, threshold=-1, hash_str="-1", 
                      surfaces=sfm.SURFACES["midthickness"]):
    """ """
    # hast_str = hash(hast_str)
    
    tmp_input_cifti = f"/tmp/wb_cifti_cluster_input_{hash_str}.dtseries.nii"
    tmp_output_cifti = f"/tmp/wb_cifti_cluster_output_{hash_str}.dtseries.nii"
    tmp_files = [tmp_input_cifti, tmp_output_cifti]
    remove_files(tmp_files)

    write_dtseries_cifti(values, tmp_input_cifti, template_cifti)
    
    left_surface, right_surface = surfaces

    cifti_cluster_command = constants.WB_COMMAND_PATH
    cifti_cluster_command += f" -cifti-find-clusters {tmp_input_cifti} {threshold} {min_size} {threshold} {min_size}"
    cifti_cluster_command += f" COLUMN {tmp_output_cifti} -left-surface {left_surface} -right-surface {right_surface} -merged-volume"
    call_res = subprocess.call(cifti_cluster_command.split())

    clustered_values = np.array(nb.load(tmp_output_cifti).get_fdata()).ravel()
    remove_files(tmp_files)
    return clustered_values[:len(values)] > 0


def spatial_filter(values, template_cifti_path, n_cpu=1, pbar=True, pbar_kwargs=dict(leave=False), n_parcels=None, **kwargs):
    """ """
    unique_parcels = np.sort(np.unique(values))
    single_parcel_set = []
    for parcel_i in unique_parcels:
        parcel_i_values = np.zeros_like(values) - 1
        parcel_i_values[values == parcel_i] = parcel_i
        single_parcel_set.append((parcel_i, parcel_i_values))

    time_str = str(time.time()).replace(".", "")
    wb_pool_func = lambda args: wb_spatial_filter(args[1], template_cifti_path, hash_str=time_str + "_" + str(args[0]))

    # for testing speed up
    if n_parcels is None:
        n_parcels = len(single_parcel_set)
    
    single_parcel_set = single_parcel_set[:n_parcels]
    n_cpu = min(os.cpu_count(), n_cpu)

    pbar_kwargs["desc"] = colored(f"Applying spatial filters to parcels", "cyan")
    pbar_kwargs["colour"] = "#008080"
    if n_cpu > 1:
        with mp.Pool(processes=n_cpu) as pool:
            if pbar:
                spatial_filter_mask_set = list(tqdm(pool.imap(wb_pool_func, single_parcel_set), total=len(single_parcel_set), **pbar_kwargs))
            else:
                spatial_filter_mask_set = pool.map(wb_pool_func, single_parcel_set)

    else:
        if pbar:
            spatial_filter_mask_set = [wb_pool_func(sp) for sp in tqdm(single_parcel_set, **pbar_kwargs)]
        else:
            spatial_filter_mask_set = map(wb_pool_func, single_parcel_set)
    
    spatial_filtered_values = np.full(len(values), fill_value=np.nan)
    for spatial_filter_mask_parcel_i, parcel_i in zip(spatial_filter_mask_set, unique_parcels):
        spatial_filtered_values[spatial_filter_mask_parcel_i] = parcel_i
    return spatial_filtered_values


def wb_dilate(values, template_cifti, hash_str=None, max_size=50, surfaces=sfm.SURFACES["midthickness"]):
    """ 'bad values' are defined as nans for the input, but wb_command uses '0' as a 'bad_value'
    """

    if isinstance(template_cifti, str):
        template_cifti = nb.load(template_cifti)

    if hash_str is None:
        has_str = hash((values.data.tobytes(), values.shape, values.dtype))

    tmp_input_cifti = f"/tmp/wb_cifti_dilate_input_{hash(hash_str)}.dtseries.nii"
    tmp_output_cifti = f"/tmp/wb_cifti_dilate_output_{hash(hash_str)}.dtseries.nii"
    tmp_files = [tmp_input_cifti, tmp_output_cifti]
    remove_files(tmp_files)

    mapped_values = values + 1
    mapped_values[np.isnan(values)] = 0
    write_dtseries_cifti(mapped_values, tmp_input_cifti, template_cifti)

    left_surface, right_surface = surfaces

    wb_dilate_command = constants.WB_COMMAND_PATH
    wb_dilate_command += f" -cifti-dilate {tmp_input_cifti} COLUMN {max_size} {max_size}"
    wb_dilate_command += f" -left-surface {left_surface} -right-surface {right_surface} {tmp_output_cifti} -nearest"

    call_res = subprocess.call(wb_dilate_command.split())
    dilated_values = np.array(nb.load(tmp_output_cifti).get_fdata()).ravel()[:len(values)] - 1
    remove_files(tmp_files)
    return dilated_values


def spatial_filter_and_dilate(partition_labels, save_path, template_cifti_path, overwrite=False, n_cpu=8, pbar=True, pbar_kwargs=dict(leave=False), min_size=50, 
                      surfaces=sfm.SURFACES["midthickness"],  **kwargs):
    """ """

    if os.path.exists(save_path) and not overwrite:
        return 

    if min_size == 0:
        partition_labels_sfd = partition_labels
    else:

        partition_labels_sf = spatial_filter(partition_labels, template_cifti_path, n_cpu=n_cpu, pbar=pbar, pbar_kwargs=pbar_kwargs,
                                             min_size=min_size, surfaces=surfaces, **kwargs)
        partition_labels_sfd = wb_dilate(partition_labels_sf, template_cifti_path, max_size=min_size, surfaces=surfaces)

    parcellate.save_partition(np.arange(len(partition_labels_sfd)), partition_labels_sfd, save_path)


def batch_spatial_filter_and_dilate(input_partition_paths, save_paths, overwrite=False,
                                    n_cpu=8, min_size=50, surfaces=sfm.SURFACES["midthickness"], pbar=True, **kwargs):
    """ """
    template_cifti = nb.load(constants.TEMPLATE_CIFTI_PATH)

    pbar_save_paths = tqdm(total=len(save_paths), desc=colored("Spatial Filtering parcels", constants.MAIN_TERM_COLOR), colour=constants.MAIN_PBAR_COLOR)
    for input_partition_path, save_path in zip(input_partition_paths, save_paths):
        partition_labels = partition_tools.load_partition_labels(input_partition_path, template_cifti)

        spatial_filter_and_dilate(partition_labels, save_path, constants.TEMPLATE_CIFTI_PATH,
                                  overwrite=overwrite, n_cpu=n_cpu, pbar=True, pbar_kwargs=dict(leave=False),
                                  min_size=min_size,
                                  surfaces=surfaces, **kwargs)
        pbar_save_paths.update(1)
    pbar_save_paths.close()


# \section end

