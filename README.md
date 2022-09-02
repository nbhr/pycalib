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

## Examples

### Single camera

1. [Intrinsic parameters from charuco images](./ipynb/incalib_charuco.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/incalib_charuco.ipynb)
   * GoPro fisheye lens distortion is handled by the rational model in OpenCV
2. [Intrinsic parameters from chessboard images](./ipynb/incalib_chess.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/incalib_chess.ipynb)
   * Zhang's method
3. [Extrinsic parameters w.r.t. a charuco board](./ipynb/excalib_charuco.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib_chess.ipynb)
   * PnP with ChAruco
4. [Extrinsic parameters w.r.t. a chassboard](./ipynb/excalib_charuco.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib_chess.ipynb)
   * PnP with chessboard
5. [Intrinsic / Extrinsic parameters from 2D-3D correspondences](./ipynb/calib2d3d.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/calib2d3d.ipynb)
   * for non-planar reference objects

### Multiple cameras

1. [Multi-view triangulation](./ipynb/ncam_triangulate.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_triangulate.ipynb)
   * DLT
2. [Sphere center detection for 2D-2D correspondences](./ipynb/sphere.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/sphere.ipynb)
   * for extrinsic calibration using a ball
3. [2-view extrinsic calibration from 2D-2D correspondences](./ipynb/excalib_2d.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib_2d.ipynb)
   * decomposes the essential matrix to R and t
4. [N-view registration](./ipynb/ncam_registration.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_registration.ipynb)
   * A linear registration of pairwise poses into a single coordinate system
5. [N-view bundle adjustment](./ipynb/ncam_ba.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_ba.ipynb)
   * A non-linear minization of reprojection errros

### 3D-3D

1. [Absolute orientation between corresponding 3D points](./ipynb/absolute_orientation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/absolute_orientation.ipynb)


## If you need to write your own calibration ...

In general, prepare some synthetic dataset, i.e., a toy example, first so that your code can return the exact solution up to the machine epsillon.  Then you can try with real data or synthetic data with noise to mimic it.

1. **Linear calibration:** Use `numpy`.
2. **Non-linear (including bundule adjustment):** Try `scipy.optimize.least_squares` first.
   * Implement your objective function as simple as possible. You do not need to consider the computational efficiency at all.
     * Test with the toy example and make sure that your objective function returns zero at the ground-truth parameter.
   * If it is unacceptably slow, try the followings in this order.
     1. Ask yourself again before trying to make it faster.  Is it really unacceptable?  If your calibration can finish in an hour and you do not do it so often, it might be OK for example. *"Premature optimization is the root of all evil."* (D. Knuth).
     2. Make sure that the optimization runs successfully anyway.  In what follows, double-check that the optimization results do not change.
     3. Vectorize the computation with `numpy`, i.e., no for-loops in the objective function.
        * or use [`numba`](https://numba.pydata.org/) (e.g. `@numba.jit`)
     4. If the system is sparse, use `jac_sparsity` option. It makes the optimization much faster even without analitical Jacobian.
     5. Implement the analytical Jacobian. You may want to use [maxima](http://wxmaxima-developers.github.io/wxmaxima/) to automate the calculation, or you may use JAX or other autodiff solutions for this.
     6. Reimplement in C++ with [ceres-solver](http://ceres-solver.org/) if the computation speed is really important.
