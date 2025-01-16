# tests/test_pm.py

import unittest
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_path + "/../")

import precision_mapping as pm

from src import constants


class TestPrecisionMapping(unittest.TestCase):

    def test_pm_arguments(self):
        args = pm.get_arguments(test_args=["-c", constants.EXAMPLE_DTSERIES, "-o tests/outputs"])
        # self.assertEqual(fmri_data[2][1], 1)

    def test_pm_main(self):
        pm.main(test_args=["-c", constants.EXAMPLE_DTSERIES, "-o", "tests/outputs", "-p", "example", "--overwrite"])
        # pm.main(test_args=["-c", constants.EXAMPLE_DTSERIES, "-o tests/outputs", "--overwrite"])


if __name__ == "__main__":
    unittest.main()
