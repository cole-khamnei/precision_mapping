# tests/test.py

import unittest
import os
import sys

# from . import constants_test_suite


TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")
import precision_mapping as pm

from precision_mapping.tests import constants_test_suite as cts


class TestPrecisionMappingCorrelators(unittest.TestCase):
    """ """

    def test_sparse_correlator_options(self):
        """ """

        default_args = dict(cifti_path=cts.TEST_DTSERIES_PATH,
                            save_path=cts.TEST_OUTPUT_VERTEX_FC_PATH,
                            censor_file=cts.TEST_DTSERIES_CENSOR_FILE,
                            block_size=1000,
                            overwrite=True,
                            sparsity=0.01,
                            backend="torch",
                            leave=True,
                            mask=pm.constants.get_geodesic_distance_mask_path(30),
                            device="cpu")

        test_function = pm.functional_connectivity.generate_correlation_matrix

        with self.subTest(test_arguments="default args - CPU test"):
            print("Testing functional_connectivity.generate_correlation_matrix:\n\tDefault args (device: cpu)")
            test_function(**default_args)

        import torch
        gpu_device = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else None

        if gpu_device:
            device_args = dict(device=gpu_device)
            with self.subTest(test_arguments=f"default args - {gpu_device} test"):
                print(f"Testing functional_connectivity.generate_correlation_matrix:\n\tdevice = {gpu_device}")
                test_function(**{**default_args, **device_args})
                print(f"{gpu_device} passed tests, using as default for future tests.")
                default_args["device"] = gpu_device

        test_arg_sets = [
            dict(backend="numpy"),
            dict(cifti_path=[cts.TEST_DTSERIES_PATH] * cts.N_TEST_REPS,
                 save_path=[cts.TEST_OUTPUT_VERTEX_FC_PATH] * cts.N_TEST_REPS,
                 censor_file=[cts.TEST_DTSERIES_CENSOR_FILE] * cts.N_TEST_REPS),
        ]
        
        for test_args in test_arg_sets:
            print(f"Testing functional_connectivity.generate_correlation_matrix:\n\tArgs: {test_args}")
            with self.subTest(test_arguments=test_args):
                test_function(**{**default_args, **test_args})


if __name__ == "__main__":
    unittest.main()
