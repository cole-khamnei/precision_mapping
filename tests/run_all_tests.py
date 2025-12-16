import os
import unittest

os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    loader = unittest.TestLoader()
    tests = loader.discover(os.path.dirname(os.path.realpath(__file__)), pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(tests)
