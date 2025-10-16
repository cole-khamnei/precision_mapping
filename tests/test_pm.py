# tests/test_pm.py

import unittest
import os
import sys

file_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir_path + "/../")

import precision_mapping as pm

from src import constants


class TestPrecisionMapping(unittest.TestCase):

    # def test_pm_arguments(self):
    #     args = pm.get_arguments(test_args=["-c", constants.EXAMPLE_DTSERIES, "-o tests/outputs"])
    #     # self.assertEqual(fmri_data[2][1], 1)

    # def test_pm_main(self):
    #     pm.main(test_args=["-c", constants.EXAMPLE_DTSERIES, "-o", "tests/outputs", "-p", "example", "--overwrite"])
    #     # pm.main(test_args=["-c", constants.EXAMPLE_DTSERIES, "-o tests/outputs", "--overwrite"])

    def test_pm_correlator(self):
        voxel_data = pm.load_voxel_data([constants.EXAMPLE_DTSERIES])
        
        sc = pm.generate_voxel_FC(voxel_data, save_path=f"{file_dir_path}/outputs/hole_test_S1.npz",
                               sparsity=0.1,
                               block_size=5000)

        sc = pm.generate_voxel_FC(voxel_data, save_path=f"{file_dir_path}/outputs/hole_test_S10.npz",
                               sparsity=1,
                               exclude_index_path=exclude_index_path,
                               mask_path=mask_path,
                               block_size=5000)

    # def test_pm_infomap(self):
    #     """ """
    #     import scipy
    #     sc_path = f"{file_dir_path}/outputs/example_voxel_FC_S1_D10_SC.npz"
    #     # sc = scipy.sparse.load_npz(sc_path)

        # pm.infomap_parallel(sc)

        # pm.infomap_parcellation(sc, save_path=f"{file_dir_path}/outputs/hole_test_S1.npy",
        #                         num_trials=1)


if __name__ == "__main__":
    unittest.main()
