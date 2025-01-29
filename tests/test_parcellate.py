# tests/test_parcellate.py

import unittest
import os
import sys
import time 

import numpy as np
import scipy

file_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir_path + "/../")

from src import parcellate


def generate_test_matrices(n, size, sparsity):
    """ """
    matrices = np.random.randn(n, size, size)
    
    thresholds = np.percentile(matrices, 100 - sparsity, axis=(1, 2), keepdims=True)
    matrices[matrices >= thresholds] = 0
    
    return matrices


class TestParcellate(unittest.TestCase):

    def test_parcellate(self):
        parcellate.main()

    def test_parallel_infomaps(self):
        """ """
        n_items = 20
        matrix_size = 1_000
        sparsity = 1
        num_trials = 20
        
        matrices = generate_test_matrices(n_items, matrix_size, sparsity)
        save_paths = [f"{file_dir_path}/outputs/test_parcel_{i}.npy" for i in range(n_items)]

        print("\tTesting parallel infomaps:")
        for n_cores in [1, 4, 8]:
            start = time.time()
            parcellate.batch_infomap_parcellation(matrices, save_paths, n_cores=n_cores, num_trials=num_trials)
            print(f"\t\t- infomaps x {n_items} with {n_cores} cores :: {time.time() - start:0.5f}s")


if __name__ == "__main__":
    unittest.main()
