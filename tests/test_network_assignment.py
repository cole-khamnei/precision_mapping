# tests/test.py

import unittest
import os
import sys

# from . import constants_test_suite

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")
import precision_mapping as pm

from precision_mapping.tests import constants_test_suite as cts


class TestNetworkAssignment(unittest.TestCase):
    """ """

    def test_network_assignment(self):
        """ """
        
        pm.utils.assert_exists(cts.TEST_SUPPLIED_PARCEL_PARTITION_PATH)
        pm.na.assign_networks_batch(cts.TEST_DTSERIES_PATH,
                                    cts.TEST_SUPPLIED_PARCEL_PARTITION_PATH,
                                    cts.TEST_OUTPUT_NETWORK_PARTITION_PATH,
                                    censor_files=cts.TEST_DTSERIES_CENSOR_FILE,
                                    overwrite=True)

if __name__ == "__main__":
    unittest.main()
