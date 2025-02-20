import argparse
import sys

from . import constants
from . import utils

sys.path.insert(0, constants.PROJECT_PATH)
import xmath_tools as xmt

# ----------------------------------------------------------------------------# 
# --------------------         Parallel Infomaps          --------------------# 
# ----------------------------------------------------------------------------# 

def get_arguments(test_args=None):
    """ """
    parser.add_argument('-c', "--ciftis", dest="cifti_txt", type=str, required=True,
                        help="Txt file with paths of cifti files")
    args = parser.parse_args() if test_args is None else parser.parse_args(test_args)


    return args


def main(test_args=None):
    """ """

    args = get_arguments(test_args=test_args)


if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
