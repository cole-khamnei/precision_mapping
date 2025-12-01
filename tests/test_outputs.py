#tests/test_outputs.py

import unittest
import os
import sys

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")
import precision_mapping as pm

from precision_mapping import partition_tools as pt
from precision_mapping.tests import constants_test_suite as cts


class TestWriteOutputs(unittest.TestCase):
    """ """

    def test_write_parcel_dlabels(self):
        """ """
        pt.write_parcel_dlabel(cts.TEST_SUPPLIED_PARCEL_PARTITION_PATH,
                               cts.TEST_OUTPUT_PARCEL_DLABEL_PATH,
                               template_cifti=cts.TEST_DTSERIES_PATH)


    def test_write_network_dlabels(self):
        """ """
        pt.write_network_dlabel(cts.TEST_SUPPLIED_NETWORK_PARTITION_PATH,
                                cts.TEST_OUTPUT_NETWORK_DLABEL_PATH,
                                template_cifti=cts.TEST_DTSERIES_PATH)

    def test_parcel_plot(self):
        """ """
        sample_label = "UNIT TEST"
        
        pm.plot.parcel_plot(cts.TEST_SUPPLIED_PARCEL_PARTITION_PATH,
                            cts.TEST_SUPPLIED_NETWORK_PARTITION_PATH,
                            sample_label,cts.TEST_OUTPUT_PARCEL_PLOT_PATH,
                            template_cifti=cts.TEST_DTSERIES_PATH)

    def test_qc_plot(self):
        """ """
        pm.plot.QC_plots(cts.TEST_SUPPLIED_PARCEL_PARTITION_PATH,
                                       cts.TEST_OUTPUT_QC_PLOT_PATH)


if __name__ == "__main__":
    unittest.main()
