import os

# \section test suite paths

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

TEST_INPUTS_DIR = f"{TEST_DIR_PATH}/inputs"
TEST_OUTPUTS_DIR = f"{TEST_DIR_PATH}/outputs"

TEST_DTSERIES_PATH = f"{TEST_INPUTS_DIR}/example_small.dtseries.nii"
TEST_DTSERIES_CENSOR_FILE = f"{TEST_INPUTS_DIR}/example_small_frame_censor.1D"

TEST_SUPPLIED_VERTEX_FC_PATH = f"{TEST_INPUTS_DIR}/example_vertex_FC.npz"
TEST_SUPPLIED_PARCEL_PARTITION_PATH = f"{TEST_INPUTS_DIR}/example_parcel_partition.npy"
TEST_SUPPLIED_NETWORK_PARTITION_PATH = f"{TEST_INPUTS_DIR}/example_network_partition.npy"

TEST_OUTPUT_VERTEX_FC_PATH = f"{TEST_OUTPUTS_DIR}/example_vertex_FC.npz"
TEST_OUTPUT_PARCEL_PARTITION_PATH = f"{TEST_OUTPUTS_DIR}/example_parcel_partition.npy"
TEST_OUTPUT_NETWORK_PARTITION_PATH = f"{TEST_OUTPUTS_DIR}/example_network_partition.npy"

TEST_OUTPUT_NETWORK_DLABEL_PATH = f"{TEST_OUTPUTS_DIR}/example_networks.dlabel.nii"
TEST_OUTPUT_PARCEL_DLABEL_PATH = f"{TEST_OUTPUTS_DIR}/example_parcels.dlabel.nii"

TEST_OUTPUT_PLOT_SAVE_PATH = f"{TEST_OUTPUTS_DIR}/example_parcellation_plot.png"

# \section test suite parameters

SKIP_DONE = True
N_TEST_REPS = 3

SKIP_FULL_PIPELINE_TEST = True

# \section end
