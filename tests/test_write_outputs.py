# tests/test.py

import unittest
import os
import sys

# from . import constants_test_suite

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")
import precision_mapping as pm

from precision_mapping.tests.constants_test_suite import *


class TestWriteOutputs(unittest.TestCase):
    """ """

    def test_write_parcel_dlabels(self):
        """ """
        pm.write.write_parcel_dlabel(TEST_SUPPLIED_PARCEL_PARTITION_PATH,
                                     TEST_OUTPUT_PARCEL_DLABEL_PATH, template_cifti=TEST_DTSERIES_PATH)


    def test_write_network_dlabels(self):
        """ """
        pm.write.write_network_dlabel(TEST_SUPPLIED_NETWORK_PARTITION_PATH,
                                      TEST_OUTPUT_NETWORK_DLABEL_PATH, template_cifti=TEST_DTSERIES_PATH)

    def test_plots(self):
        """ """
        sample_label = "UNIT TEST"
        
        pm.write.parcel_plot(TEST_SUPPLIED_PARCEL_PARTITION_PATH, TEST_SUPPLIED_NETWORK_PARTITION_PATH,
                             sample_label,TEST_OUTPUT_PLOT_SAVE_PATH, template_cifti=TEST_DTSERIES_PATH)


if __name__ == "__main__":
    unittest.main()
