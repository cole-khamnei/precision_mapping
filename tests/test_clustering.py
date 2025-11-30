# tests/test.py

import unittest
import os
import sys

# from . import constants_test_suite

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")
import precision_mapping as pm

from precision_mapping.tests.constants_test_suite import *


class TestParcelClustering(unittest.TestCase):
    """ """

    def test_parcel_detection(self):
        """ infomaps part"""

        pm.utils.assert_exists(TEST_SUPPLIED_VERTEX_FC_PATH)
        pm.parcellate.parcel_detection(TEST_SUPPLIED_VERTEX_FC_PATH,
                                       TEST_OUTPUT_PARCEL_PARTITION_PATH,
                                       n_cores=1, n_reps=1,
                                       overwrite=True, silent=True)

if __name__ == "__main__":
    unittest.main()
