import unittest
from testbook import testbook

import numpy as np
import os
import cv2
import pycalib

# https://qiita.com/Syuparn/items/fbb8533e5f4fa3181ffd
def local_chdir(func):
    def _inner(*args, **kwargs):
        # get the current dir before entering the func
        dir_original = os.getcwd()
        # run func
        ret = func(*args, **kwargs)
        # back to the original dir
        os.chdir(dir_original)
        return ret
    return _inner

class TestPyCalib(unittest.TestCase):
    """
    verifies that all notebooks run without error
    """

    @classmethod
    def setUpClass(cls):
        pass

    @local_chdir
    def run_ipynb(self, file):
        os.chdir('./ipynb')
        with testbook(file, execute=True) as tb:
            pass

    def test_absolute_orientation(self):
        self.run_ipynb('absolute_orientation.ipynb')

    def test_absolute_orientation(self):
        self.run_ipynb('aruco_movie.ipynb')

    def test_calib2d3d(self):
        self.run_ipynb('calib2d3d.ipynb')

    def test_charuco_diamond(self):
        self.run_ipynb('charuco_diamond.ipynb')

    def test_excalib_2d(self):
        self.run_ipynb('excalib_2d.ipynb')

    def test_excalib_chess(self):
        self.run_ipynb('excalib_chess.ipynb')

    def test_incalib_charuco(self):
        self.run_ipynb('incalib_charuco.ipynb')

    def test_incalib_chess(self):
        self.run_ipynb('incalib_chess.ipynb')

    def test_ncam_ba(self):
        self.run_ipynb('ncam_ba.ipynb')

    def test_ncam_registration(self):
        self.run_ipynb('ncam_registration.ipynb')

    def test_ncam_triangulate(self):
        self.run_ipynb('ncam_triangulate.ipynb')

    def test_example_gopro_ba(self):
        self.run_ipynb('example_gopro_ba.ipynb')

    def test_excalib_charuco(self):
        self.run_ipynb('excalib_charuco.ipynb')

    def test_sphere(self):
        self.run_ipynb('sphere.ipynb')
