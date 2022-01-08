# Simple Camera Calibration in Python for Beginners

This is a collection of algorithms related to multiple-view camera calibration in computer vision.  Please note that the goal of this package is to provide *minimal* examples to demostrate the concept for beginners (i.e., students).  For large-scale, realtime, accurate, robust, production-quality implementations, or for implementations for your specific situation, please consult your advisor.


## Disclaimer

This is research software and may contain bugs or other issues -- please use it at your own risk. If you experience major problems with it, you may contact us, but please note that we do not have the resources to deal with all issues.

## How to use

You can simply install the package by `pip`.
```
$ python3 -m pip install -U pycalib-simple
```

The pip installation, however, does not include examples in `./ipynb`.  To run examples, download the repository explicitly.  For example,
1. **Local:** You can clone/download this repository to your local PC, and open `./ipynb/*.ipynb` files by your local Jupyter.
2. **Colaboratory:** You can open each Jupyter notebook directly in Google Colaboratory by clicking the ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) buttons below.
   * *Warning:* Most of them do not run properly as-is, since colab does not clone images used in the Jupyter notebooks. Please upload required files manually. (or run `!pip install` and `!git clone` at the beginning of each notebook.)

## Single camera

1. [Intrinsic parameters from chessboard images](./ipynb/incalib.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/incalib.ipynb)
2. [Extrinsic parameters w.r.t. a chassboard](./ipynb/excalib-chess.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib-chess.ipynb)
3. [Intrinsic / Extrinsic parameters from 2D-3D correspondences](./ipynb/calib2d3d.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/calib2d3d.ipynb)
4. [Distortion](./ipynb/distortion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/distortion.ipynb)

## Multiple cameras

1. [Sphere center detection for 2D-2D correspondences](./ipynb/sphere.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/sphere.ipynb)
1. [2-view extrinsic calibration from 2D-2D correspondences](./ipynb/excalib-2d.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib-2d.ipynb)
2. [N-view registration](./ipynb/ncam_registration.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_registration.ipynb)
3. [N-view bundle adjustment](./ipynb/ncam_ba.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_ba.ipynb)


## 3D-3D

1. [Absolute orientation between corresponding 3D points](./ipynb/absolute_orientation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/absolute_orientation.ipynb)


## If you need to write your own calibration ...

1. For linear case:
   * Use `numpy`.
2. For non-linear (including bundule adjustment) case
   1. Try `scipy.optimize.least_squares` first.
      * If the system is sparse, use `jac_sparsity` option. It makes the optimization much faster even without analitical Jacobian.
      * If it is slow, use `numba`.
   2. Use [ceres-solver](http://ceres-solver.org/) if the computation speed is really important.
      * Make sure the optimization is doable with `scipy` first.
