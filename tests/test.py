# tests/test.py

import unittest
import os
import sys

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")

import precision_mapping as pm

# from src import constants

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

# TODO: move to constants? break out tests into multiple scripts?

SKIP_DONE = True


class TestPrecisionMapping(unittest.TestCase):
    """ """


    def test_sparse_correlator_options(self):
        """ """
        # if SKIP_DONE:
        #     return

        N_TEST_REPS = 3

        default_args = dict(cifti_path=TEST_DTSERIES_PATH,
                            save_path=TEST_OUTPUT_VERTEX_FC_PATH,
                            censor_file=TEST_DTSERIES_CENSOR_FILE,
                            block_size=1000,
                            overwrite=True,
                            backend="torch",
                            device="cpu")
        test_function = pm.functional_connectivity.generate_correlation_matrix

        with self.subTest(test_arguments="default args - CPU test"):
            print(f"Testing functional_connectivity.generate_correlation_matrix:\n\tDefault args (device: cpu)")
            test_function(**default_args)

        import torch
        gpu_device = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else None

        if gpu_device:
            device_args = dict(device=gpu_device)
            with self.subTest(test_arguments=f"default args - {gpu_device} test"):
                print(f"Testing functional_connectivity.generate_correlation_matrix:\n\tdevice = {gpu_device}")
                test_function(**{**default_args, **device_args})
                print(f"{gpu_device} passed tests, using as default for future tests.")
                default_args["device"] = gpu_device

        test_arg_sets = [
            dict(backend="numpy"),
            dict(cifti_path=[TEST_DTSERIES_PATH] * N_TEST_REPS,
                 save_path=[TEST_OUTPUT_VERTEX_FC_PATH] * N_TEST_REPS,
                 censor_file=[TEST_DTSERIES_CENSOR_FILE] * N_TEST_REPS),
        ]
        
        for test_args in test_arg_sets:
            print(f"Testing functional_connectivity.generate_correlation_matrix:\n\tArgs: {test_args}")
            with self.subTest(test_arguments=test_args):
                test_function(**{**default_args, **test_args})


    def test_sparse_correlator(self):
        """ """
        if SKIP_DONE:
            return

        sc = pm.functional_connectivity.generate_correlation_matrix(TEST_DTSERIES_PATH,
                                                                    TEST_OUTPUT_VERTEX_FC_PATH,
                                                                    censor_file=TEST_DTSERIES_CENSOR_FILE,
                                                                    block_size=1000, overwrite=True,
                                                                    backend="torch", device="cpu")

    def test_parcel_detection(self):
        """ infomaps part"""

        if SKIP_DONE:
            return

        pm.utils.assert_exists(TEST_SUPPLIED_VERTEX_FC_PATH)
        pm.parcellate.parcel_detection(TEST_SUPPLIED_VERTEX_FC_PATH,
                                       TEST_OUTPUT_PARCEL_PARTITION_PATH,
                                       n_cores=1, n_reps=1,
                                       overwrite=True, silent=True)

    def test_network_assignment(self):
        """ """
        if SKIP_DONE:
            return

        pm.utils.assert_exists(TEST_SUPPLIED_PARCEL_PARTITION_PATH)
        pm.na.assign_networks_batch(TEST_DTSERIES_PATH,
                                    TEST_SUPPLIED_PARCEL_PARTITION_PATH,
                                    TEST_OUTPUT_NETWORK_PARTITION_PATH,
                                    censor_files=TEST_DTSERIES_CENSOR_FILE,
                                    overwrite=True)


    def test_write_parcel_dlabels(self):
        """ """
        if SKIP_DONE:
            return
        
        pm.write.write_parcel_dlabel(TEST_SUPPLIED_PARCEL_PARTITION_PATH,
                                     TEST_OUTPUT_PARCEL_DLABEL_PATH, template_cifti=TEST_DTSERIES_PATH)


    def test_write_network_dlabels(self):
        """ """
        if SKIP_DONE:
            return
        
        pm.write.write_network_dlabel(TEST_SUPPLIED_NETWORK_PARTITION_PATH,
                                      TEST_OUTPUT_NETWORK_DLABEL_PATH, template_cifti=TEST_DTSERIES_PATH)

    def test_plots(self):
        """ """
        if SKIP_DONE:
            return
        
        sample_label = "UNIT TEST"
        
        pm.write.parcel_plot(TEST_SUPPLIED_PARCEL_PARTITION_PATH, TEST_SUPPLIED_NETWORK_PARTITION_PATH,
                             sample_label,TEST_OUTPUT_PLOT_SAVE_PATH, template_cifti=TEST_DTSERIES_PATH)


    def test_path_handling(self):
        """ """

        if SKIP_DONE:
            return
        
        subject_ids = "outputs"
        sample_labels="example"

        path_sets = pm.utils.create_pm_paths(subject_ids, sample_labels, TEST_DIR_PATH)
        (vertex_fc_paths, parcel_partition_paths, network_partition_paths, 
         parcel_dlabel_paths, network_dlabel_paths, plot_save_paths) = path_sets

        self.assertEqual(vertex_fc_paths, [TEST_OUTPUT_VERTEX_FC_PATH])
        self.assertEqual(parcel_partition_paths, [TEST_OUTPUT_PARCEL_PARTITION_PATH])
        self.assertEqual(network_partition_paths, [TEST_OUTPUT_NETWORK_PARTITION_PATH])
        self.assertEqual(parcel_dlabel_paths, [TEST_OUTPUT_PARCEL_DLABEL_PATH])
        self.assertEqual(network_dlabel_paths, [TEST_OUTPUT_NETWORK_DLABEL_PATH])
        self.assertEqual(plot_save_paths, [TEST_OUTPUT_PLOT_SAVE_PATH])


    def test_full_pipeline(self):
        """ """
        if SKIP_DONE:
            return

        subject_ids = "outputs"
        sample_labels="example"

        pm.precision_mapping(TEST_DTSERIES_PATH, subject_ids, sample_labels, TEST_DIR_PATH,
                             overwrite=True, device="cpu", n_cores=1, n_infomaps_reps=1)


    def test_load_voxel_data(self):
        """ """
        if SKIP_DONE:
            return

        voxel_data = pm.utils.load_voxel_data(TEST_DTSERIES_PATH)
        self.assertEqual(voxel_data.shape, (876, 91282))
        censored_voxel_data = pm.utils.load_voxel_data(TEST_DTSERIES_PATH, censor_file=TEST_DTSERIES_CENSOR_FILE)
        self.assertEqual(censored_voxel_data.shape, (824, 91282))
        

    def test_pm_arguments(self):
        """ """
        if SKIP_DONE:
            return
        subject_ids = "outputs"
        sample_labels="example"

        args = pm.get_arguments(test_args=["-c", TEST_DTSERIES_PATH, "-o", TEST_DIR_PATH,
                                           "-i", subject_ids, "-l", sample_labels,
                                           "--censor-file", TEST_DTSERIES_CENSOR_FILE])

    def test_main(self):
        """ """
        if SKIP_DONE:
            return

        subject_ids = "outputs"
        sample_labels="example"

        args = pm.main(test_args=["-c", TEST_DTSERIES_PATH, "-o", TEST_DIR_PATH,
                                  "-i", subject_ids, "-l", sample_labels])


if __name__ == "__main__":
    unittest.main()
