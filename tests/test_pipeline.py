# tests/test.py

import unittest
import os
import sys

# from . import constants_test_suite

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")
import precision_mapping as pm

from precision_mapping.tests.constants_test_suite import *


class TestPipeline(unittest.TestCase):
    """ """

    def test_path_handling(self):
        """ """

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

    def test_load_voxel_data(self):
        """ """

        voxel_data = pm.utils.load_voxel_data(TEST_DTSERIES_PATH)
        self.assertEqual(voxel_data.shape, (30, 91282))
        censored_voxel_data = pm.utils.load_voxel_data(TEST_DTSERIES_PATH, censor_file=TEST_DTSERIES_CENSOR_FILE)
        self.assertEqual(censored_voxel_data.shape, (28, 91282))
        

    def test_pm_arguments(self):
        """ """

        subject_ids = "outputs"
        sample_labels="example"

        args = pm.get_arguments(test_args=["-c", TEST_DTSERIES_PATH, "-o", TEST_DIR_PATH,
                                           "-i", subject_ids, "-l", sample_labels,
                                           "--censor-file", TEST_DTSERIES_CENSOR_FILE])

    def test_full_pipeline(self):
        """ """
        if SKIP_FULL_PIPELINE_TEST:
            return

        subject_ids = "outputs"
        sample_labels="example"

        pm.precision_mapping(TEST_DTSERIES_PATH, subject_ids, sample_labels, TEST_DIR_PATH,
                             overwrite=True, device="cpu", n_cores=1, n_infomaps_reps=1)


    def test_main(self):
        """ """
        if SKIP_FULL_PIPELINE_TEST:
            return

        subject_ids = "outputs"
        sample_labels="example"

        args = pm.main(test_args=["-c", TEST_DTSERIES_PATH, "-o", TEST_DIR_PATH,
                                  "-i", subject_ids, "-l", sample_labels])


if __name__ == "__main__":
    unittest.main()
