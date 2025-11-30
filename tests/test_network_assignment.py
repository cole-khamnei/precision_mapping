# tests/test.py

import unittest
import os
import sys

# from . import constants_test_suite

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")
import precision_mapping as pm

from precision_mapping.tests.constants_test_suite import *


class TestNetworkAssignment(unittest.TestCase):
    """ """

    def test_network_assignment(self):
        """ """
        
        pm.utils.assert_exists(TEST_SUPPLIED_PARCEL_PARTITION_PATH)
        pm.na.assign_networks_batch(TEST_DTSERIES_PATH,
                                    TEST_SUPPLIED_PARCEL_PARTITION_PATH,
                                    TEST_OUTPUT_NETWORK_PARTITION_PATH,
                                    censor_files=TEST_DTSERIES_CENSOR_FILE,
                                    overwrite=True)

if __name__ == "__main__":
    unittest.main()
