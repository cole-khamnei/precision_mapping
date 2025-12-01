# tests/test.py

import unittest
import os
import sys

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")
import precision_mapping as pm

from precision_mapping.tests import constants_test_suite as cts


class TestParcelClustering(unittest.TestCase):
    """ """

    def test_parcel_detection(self):
        """ infomaps part"""

        pm.utils.assert_exists(cts.TEST_SUPPLIED_VERTEX_FC_PATH)
        pm.parcellate.parcel_detection(cts.TEST_SUPPLIED_VERTEX_FC_PATH,
                                       cts.TEST_OUTPUT_PARCEL_PARTITION_PATH,
                                       n_cores=1, n_reps=1,
                                       overwrite=True, silent=True)

if __name__ == "__main__":
    unittest.main()
