import numpy as np
import nibabel as nb

from . import constants

# ----------------------------------------------------------------------------# 
# --------------------           Cifti Helpers            --------------------# 
# ----------------------------------------------------------------------------# 


def read_censor_file(censor_file):
    """ """
    with open(censor_file, 'r') as file:
        censor_data = np.array(file.read().strip().split()).astype(float) == 1

    return censor_data


def load_voxel_data(dtseries_paths, censor_file=None, dtype="float32"):
    """ """

    if not isinstance(dtseries_paths, str):
        return np.vstack([load_voxel_data(path) for path in dtseries_paths])

    assert (dtseries_paths.endswith(".npy") or dtseries_paths.endswith(".nii"))

    if dtseries_paths.endswith(".npy"):
        voxel_data = np.load(dtseries_paths).astype(dtype)
    else:
        cifti = nb.load(dtseries_paths)
        voxel_data = cifti.get_fdata(caching="unchanged", dtype=dtype)

    if censor_file is not None:
        dat_censor_indices = read_censor_file(censor_file)
        voxel_data = voxel_data[dat_censor_indices]

    return voxel_data


def get_template_cifti(template_cifti=None):
    """ """
    if template_cifti is None:
        return nb.load(constants.TEMPLATE_CIFTI_PATH)

    if isinstance(template_cifti, str):
        template_cifti = nb.load(template_cifti)

    return template_cifti


def cifti_map(rois, roi_values, template_cifti, fill_value=np.nan):
    """ """

    if rois is not None:
        assert roi_values.shape[-1] <= len(rois)

    pax = template_cifti.header.get_axis(1)
    prefix_shape = roi_values.shape[:-1] if roi_values.ndim > 1 else ()
    lh_values = np.full(shape=(pax.nvertices["CIFTI_STRUCTURE_CORTEX_LEFT"], *prefix_shape), fill_value=fill_value)
    rh_values = np.full(shape=(pax.nvertices["CIFTI_STRUCTURE_CORTEX_RIGHT"], *prefix_shape), fill_value=fill_value)

    if roi_values.ndim > 1:
        roi_values = roi_values.T
    # create plot values dict from template cifti
    if isinstance(pax, nb.cifti2.ParcelsAxis):
        for roi_value, roi in zip(roi_values, rois):
            _, kld = pax[roi]
            lh_values[kld.get("CIFTI_STRUCTURE_CORTEX_LEFT", [])] = roi_value
            rh_values[kld.get("CIFTI_STRUCTURE_CORTEX_RIGHT", [])] = roi_value

    elif isinstance(pax, nb.cifti2.BrainModelAxis):
        slice_LUT = {structure: sl for structure, sl,_  in pax.iter_structures()}
        lh_indices, rh_indices = [], []

        for i in range(len(pax)):
            _, ind, structure = pax[i]
            if "CORTEX_LEFT" in structure:
                lh_indices.append(ind)
            if "CORTEX_RIGHT" in structure:
                rh_indices.append(ind)

        lh_indices = np.array(lh_indices)
        rh_indices = np.array(rh_indices)

        lh_values[lh_indices] = roi_values[slice_LUT["CIFTI_STRUCTURE_CORTEX_LEFT"]]
        rh_values[rh_indices] = roi_values[slice_LUT["CIFTI_STRUCTURE_CORTEX_RIGHT"]]

    return {"left": np.moveaxis(lh_values, 0, -1), "right": np.moveaxis(rh_values, 0, -1)}


def get_cortex_data(full_data, cifti):
    """ """
    pax = cifti.header.get_axis(1)
    slice_LUT = {structure: sl for structure, sl,_  in pax.iter_structures()}
    cortex_data_L = full_data[:, slice_LUT["CIFTI_STRUCTURE_CORTEX_LEFT"]]
    cortex_data_R = full_data[:, slice_LUT["CIFTI_STRUCTURE_CORTEX_RIGHT"]]
    return np.hstack([cortex_data_L, cortex_data_R])


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
