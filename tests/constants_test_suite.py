import os

# ----------------------------------------------------------------------------#
# ----------------           Test Suite Parameters            ----------------#
# ----------------------------------------------------------------------------#

N_TEST_REPS = 3

SKIP_FULL_PIPELINE_TEST = True

# ----------------------------------------------------------------------------# 
# --------------------          Test Suite Paths          --------------------# 
# ----------------------------------------------------------------------------# 

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_INPUTS_DIR = f"{TEST_DIR_PATH}/inputs"
TEST_OUTPUTS_DIR = f"{TEST_DIR_PATH}/outputs"

#input paths
TEST_DTSERIES_PATH = f"{TEST_INPUTS_DIR}/example_small.dtseries.nii"
TEST_DTSERIES_CENSOR_FILE = f"{TEST_INPUTS_DIR}/example_small_frame_censor.dat"
TEST_SUBJECT_ID = "outputs"
TEST_SAMPLE_LABEL = "example"

#txt file inputs
TEST_DTSERIES_LIST_PATH = f"{TEST_INPUTS_DIR}/example_small_dtseries_list.txt"
TEST_DTSERIES_CENSOR_FILE_LIST_PATH = f"{TEST_INPUTS_DIR}/example_small_frame_censor_list.txt"
TEST_SUBJECT_IDS_LIST_PATH = f"{TEST_INPUTS_DIR}/example_small_subject_ids_list.txt"
TEST_SAMPLE_LABELS_LIST_PATH = f"{TEST_INPUTS_DIR}/example_small_sample_labels_list.txt"

# supplied intermediates
TEST_SUPPLIED_VERTEX_FC_PATH = f"{TEST_INPUTS_DIR}/example_vertex_FC.npz"
TEST_SUPPLIED_PARCEL_PARTITION_PATH = f"{TEST_INPUTS_DIR}/example_parcel_partition.npy"
TEST_SUPPLIED_NETWORK_PARTITION_PATH = f"{TEST_INPUTS_DIR}/example_network_partition.npy"

#output paths
TEST_OUTPUT_VERTEX_FC_PATH = f"{TEST_OUTPUTS_DIR}/example_vertex_FC.npz"
TEST_OUTPUT_PARCEL_PARTITION_PATH = f"{TEST_OUTPUTS_DIR}/example_parcel_partition.npy"
TEST_OUTPUT_NETWORK_PARTITION_PATH = f"{TEST_OUTPUTS_DIR}/example_network_partition.npy"
TEST_OUTPUT_NETWORK_DLABEL_PATH = f"{TEST_OUTPUTS_DIR}/example_networks.dlabel.nii"
TEST_OUTPUT_PARCEL_DLABEL_PATH = f"{TEST_OUTPUTS_DIR}/example_parcels.dlabel.nii"
TEST_OUTPUT_PARCEL_PLOT_PATH = f"{TEST_OUTPUTS_DIR}/example_parcellation_plot.png"
TEST_OUTPUT_QC_PLOT_PATH = f"{TEST_OUTPUTS_DIR}/example_QC_plot.png"

#generating txt files
def write_txt_file(txt_file, input_item, n_lines):
    """ """
    with open(txt_file, 'w') as file:
        file.write("\n".join([input_item] * n_lines))

write_txt_file(TEST_DTSERIES_LIST_PATH, TEST_DTSERIES_PATH, N_TEST_REPS)
write_txt_file(TEST_DTSERIES_CENSOR_FILE_LIST_PATH, TEST_DTSERIES_CENSOR_FILE, N_TEST_REPS)
write_txt_file(TEST_SUBJECT_IDS_LIST_PATH, TEST_SUBJECT_ID, N_TEST_REPS)
write_txt_file(TEST_SAMPLE_LABELS_LIST_PATH, TEST_SAMPLE_LABEL, N_TEST_REPS)

# \section constant values

FULL_CIFTI_N_VERTEX = 91_282

# ----------------------------------------------------------------------------#
# --------------------                End                 --------------------#
# ----------------------------------------------------------------------------#
