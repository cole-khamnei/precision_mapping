#tests/test_pipeline.py

import unittest
import os
import sys

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")
import precision_mapping as pm

from precision_mapping.tests import constants_test_suite as cts


class TestPipeline(unittest.TestCase):
    """ """

    def test_path_handling(self):
        """ """

        paths = pm.utils.create_pm_paths(cts.TEST_SUBJECT_ID, cts.TEST_SAMPLE_LABEL,
                                         TEST_DIR_PATH)

        self.assertEqual(paths["vertex-fc"], [cts.TEST_OUTPUT_VERTEX_FC_PATH])
        self.assertEqual(paths["parcel-partition"], [cts.TEST_OUTPUT_PARCEL_PARTITION_PATH])
        self.assertEqual(paths["network-partition"], [cts.TEST_OUTPUT_NETWORK_PARTITION_PATH])
        self.assertEqual(paths["parcel-dlabel"], [cts.TEST_OUTPUT_PARCEL_DLABEL_PATH])
        self.assertEqual(paths["network-dlabel"], [cts.TEST_OUTPUT_NETWORK_DLABEL_PATH])
        self.assertEqual(paths["parcel-plot"], [cts.TEST_OUTPUT_PARCEL_PLOT_PATH])
        self.assertEqual(paths["qc-plot"], [cts.TEST_OUTPUT_QC_PLOT_PATH])

    def test_load_voxel_data(self):
        """ """
        voxel_data = pm.cifti_tools.load_voxel_data(cts.TEST_DTSERIES_PATH)
        self.assertEqual(voxel_data.shape, (30, cts.FULL_CIFTI_N_VERTEX))
        censored_voxel_data = pm.cifti_tools.load_voxel_data(cts.TEST_DTSERIES_PATH,
                                                             censor_file=cts.TEST_DTSERIES_CENSOR_FILE)
        self.assertEqual(censored_voxel_data.shape, (29, cts.FULL_CIFTI_N_VERTEX))

    def test_pm_arguments(self):
        """ """
        pm.get_arguments(test_args=["-c", cts.TEST_DTSERIES_PATH, "-o", TEST_DIR_PATH,
                                    "-i", cts.TEST_SUBJECT_ID,
                                    "-l", cts.TEST_SAMPLE_LABEL,
                                    "--censor-file", cts.TEST_DTSERIES_CENSOR_FILE])


    def test_pm_txt_arguments(self):
        """ """
        pm.get_arguments(test_args=["-c", cts.TEST_DTSERIES_LIST_PATH,
                                    "-o", TEST_DIR_PATH,
                                    "-i", cts.TEST_SUBJECT_IDS_LIST_PATH,
                                    "-l", cts.TEST_SAMPLE_LABELS_LIST_PATH,
                                    "--censor-file", cts.TEST_DTSERIES_CENSOR_FILE_LIST_PATH])

    def test_full_pipeline(self):
        """ """
        if cts.SKIP_FULL_PIPELINE_TEST:
            return

        pm.precision_mapping(cts.TEST_DTSERIES_PATH, cts.TEST_SUBJECT_ID,
                             cts.TEST_SAMPLE_LABEL, TEST_DIR_PATH,
                             sparsity=0.01, overwrite=True,
                             n_cores=1, n_infomaps_reps=1)

    def test_main_single_input(self):
        # """ """
        # if cts.SKIP_FULL_PIPELINE_TEST:
        #     return

        pm.main(test_args=["-c", cts.TEST_DTSERIES_PATH,
                           "-o", TEST_DIR_PATH,
                           "-i", cts.TEST_SUBJECT_ID,
                           "-l", cts.TEST_SAMPLE_LABEL,
                           # "--overwrite", 
                           "--spatial-filter-n-parcels", "10",
                           "--spatial-filter-size", "1",
                           "--n-reps", "1", "--sparsity", "0.01"])

    def test_main_txt_input(self):
        """ """
        if cts.SKIP_FULL_PIPELINE_TEST:
            return

        pm.main(test_args=["-c", cts.TEST_DTSERIES_LIST_PATH,
                           "-o", TEST_DIR_PATH,
                           "-i", cts.TEST_SUBJECT_IDS_LIST_PATH,
                           "-l", cts.TEST_SAMPLE_LABELS_LIST_PATH,
                           "--censor-file", cts.TEST_DTSERIES_CENSOR_FILE_LIST_PATH,
                           "--overwrite",
                           "--n-reps", "1", "--sparsity", "0.01"])


if __name__ == "__main__":
    unittest.main()
