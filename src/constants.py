# ----------------------------------------------------------------------------# 
# --------------------               Paths                --------------------# 
# ----------------------------------------------------------------------------# 

EC_DIR = "/data/data4/earlycort_7114"

EXAMPLE_SUBJECT = "ec_08_0050"
EC_FILE_EXT = "Atlas_mov_regr_b_lt_resid_brain_regr_resid_DV_FDadapt_frcintrp_bpss_lc2353_hc2_fwhm4"

VALID_EC_SESSIONS = ["RS_fMRI_maps"]


def get_EC_cifti_path(subject, session, file_type, file_ext=EC_FILE_EXT):
    """ """
    assert session in VALID_EC_SESSIONS
    return f"{EC_DIR}/{subject}/MNINonLinear/Results/{session}/{session}_{file_ext}.{file_type}.nii"

EXAMPLE_DTSERIES = get_EC_cifti_path(EXAMPLE_SUBJECT, "RS_fMRI_maps", "dtseries")

# print(EXAMPLE_DTSERIES)
