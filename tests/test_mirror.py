import unittest
import numpy as np
import cv2
import pycalib

class TestPyCalibMirror(unittest.TestCase):
    def prepare(self):
        DATA_DIR = './data/mirror/chess/'
        model = np.loadtxt(f'{DATA_DIR}/model.txt')
        #print(f'3D points (#_of_reference_points x 3): {model.shape}')

        input1 = np.loadtxt(f'{DATA_DIR}/input1.txt')
        input2 = np.loadtxt(f'{DATA_DIR}/input2.txt')
        input3 = np.loadtxt(f'{DATA_DIR}/input3.txt')
        input4 = np.loadtxt(f'{DATA_DIR}/input4.txt')
        input5 = np.loadtxt(f'{DATA_DIR}/input5.txt')
        input = np.array([input1, input2, input3, input4, input5])
        #print(f'2D points (#_of_mirrors x #_of_reference_points x 2)): {input.shape}')

        K = np.loadtxt(f'{DATA_DIR}/camera.txt')
        #print(f'K = {K}')

        return model, input, K

    def test_mirror_5(self):
        model, input, K = self.prepare()

        # linear solution
        R0, T0, n0, d0, rep0 = pycalib.mirror.tnm(model, input, K)
        #print(f'Reprojection error (linear) = {rep0:.3} px')
        self.assertLess(rep0, 6.284716397404586)

        # non-linear refinement
        R, T, n, d, rep = pycalib.mirror.tnm_ba(model, input, K, R0, T0, n0, d0)
        #print(f'Reprojection error (BA) = {rep:.3} px')
        self.assertLess(rep, 0.6401348975172396)

    def test_mirror_3(self):
        model, input, K = self.prepare()
        input = input[:3]

        # linear solution
        R0, T0, n0, d0, rep0 = pycalib.mirror.tnm(model, input, K)
        #print(f'Reprojection error (linear) = {rep0:.3} px')
        self.assertLess(rep0, 1.51)

        # non-linear refinement
        R, T, n, d, rep = pycalib.mirror.tnm_ba(model, input, K, R0, T0, n0, d0)
        #print(f'Reprojection error (BA) = {rep:.3} px')
        self.assertLess(rep, 0.689)

    def test_mirror_mask(self):
        model, input, K = self.prepare()

        TH = 25
        rng = np.random.default_rng(0)
        for x in input:
            sign = rng.integers(0, 100, len(x))
            x[sign < TH, :] = np.nan
        # print(x)

        # print(f'{TH}% of observations are randomly masked')

        # linear solution
        R0, T0, n0, d0, rep0 = pycalib.mirror.tnm(model, input, K)
        #print(f'Reprojection error (linear) = {rep0:.3} px')
        self.assertLess(rep0, 6.493891993463244)

        # non-linear refinement
        R, T, n, d, rep = pycalib.mirror.tnm_ba(model, input, K, R0, T0, n0, d0)
        #print(f'Reprojection error (BA) = {rep:.3} px')
        self.assertLess(rep, 0.6538855436697778)
