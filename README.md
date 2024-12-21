# Simple Camera Calibration in Python for Beginners

This is a collection of algorithms related to multiple view camera calibration in computer vision.  Please note that the goal of this package is to provide *minimal* examples to demostrate the concept for beginners (i.e., students).  For large-scale, realtime, accurate, robust, production-quality implementations, or for implementations for your specific situation, please consult your advisor.


## Disclaimer

This is research software and may contain bugs or other issues -- please use it at your own risk. If you experience major problems with it, you may contact us, but please note that we do not have the resources to deal with all issues.

## How to use

You can simply install the package by `pip`.

```:bash
python3 -m pip install -U pycalib-simple
```

The pip installation, however, does not include examples in [`./ipynb/`](./ipynb/) or tools in [`./tools/`](./tools/).  To run examples and tools, download the repository explicitly.  For example,

1. **Local:** You can clone/download this repository to your local PC, and open `./ipynb/*.ipynb` files by your local Jupyter.
2. **Colaboratory:** You can open each Jupyter notebook directly in Google Colaboratory by clicking the ![Open In Colab][def] buttons below.
   * *Warning:* Most of them do not run properly as-is, since colab does not clone images used in the Jupyter notebooks. Please upload required files manually. (or run `!pip install` and `!git clone` at the beginning of each notebook.)

Notice that the scripts in `./tools/` are not supposed to run in Colab/Jupyter.


## Examples

1. [Extrinsic calibration of 15 GoPro cameras](./ipynb/example_gopro_ba.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/example_gopro_ba.ipynb)
   * Given 2D-2D correspondences, this example calibrates the extrinsic parameters of 15 GoPro cams.

### Single camera

1. [Intrinsic calibration with charuco images](./ipynb/incalib_charuco.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/incalib_charuco.ipynb)
   * GoPro fisheye lens distortion is handled by the rational model in OpenCV
2. [Intrinsic calibration with chessboard images](./ipynb/incalib_chess.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/incalib_chess.ipynb)
   * Zhang's method
3. [Extrinsic calibration w.r.t. a charuco board](./ipynb/excalib_charuco.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib_charuco.ipynb)
   * PnP with ChAruco
4. [Extrinsic calibration w.r.t. a chessboard](./ipynb/excalib_chess.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib_chess.ipynb)
   * PnP with chessboard
5. [Intrinsic / Extrinsic calibration with 2D-3D correspondences](./ipynb/calib2d3d.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/calib2d3d.ipynb)
   * for non-planar reference objects

### Multiple cameras

1. [Multi-view triangulation](./ipynb/ncam_triangulate.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_triangulate.ipynb)
   * N-view DLT
2. [ChAruco diamond marker detection for 2D-2D correspondences](./ipynb/charuco_diamond.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/charuco_diamond.ipynb)
   * for extrinsic calibration using a ChAruco diamond marker
   * also can be used for PnP, i.e., extrinsic calibration w.r.t. the diamond marker
3. [Sphere center detection for 2D-2D correspondences](./ipynb/sphere.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/sphere.ipynb)
   * for extrinsic calibration using a ball
   * A color-based ball detection is provided as [`tools/detect_by_color_gui.py`](tools/detect_by_color_gui.py) and [`tools/detect_by_color.py`](tools/detect_by_color.py).  The former GUI version can be used to sample foreground and background pixel colors, and the latter can be used to process each frame.
4. [2-view extrinsic calibration from 2D-2D correspondences](./ipynb/excalib_2d.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib_2d.ipynb)
   * decomposes the essential matrix to R and t
5. [N-view registration](./ipynb/ncam_registration.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_registration.ipynb)
   * A linear registration of pairwise poses into a single coordinate system
6. [N-view bundle adjustment](./ipynb/ncam_ba.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_ba.ipynb)
   * A non-linear minization of reprojection errors
7. [N-view time sync](./ipynb/qrtimecode.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/qrtimecode.ipynb)
   * GoPro compatible QR time sync pattern generator, detector, and offset estimator

### 3D-3D

1. [Absolute orientation (or similarity transform) between corresponding 3D points](./ipynb/absolute_orientation.ipynb) [![Open In Colab][def]](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/absolute_orientation.ipynb)


## If you need to write your own calibration ...

In general, prepare some synthetic dataset, i.e., a toy example, first so that your code can return the exact solution up to the machine epsillon.  Then you can try with real data or synthetic data with noise to mimic it.

Also you may want to read Section A6.3 "A sparse Levenberg-Marquardt algorithm" of the textbook "Multiple View Geometry in Computer Vision" by Hartley and Zisserman.

1. **Linear calibration:** Use `numpy`.
2. **Non-linear (including bundule adjustment):** Try [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) first.
   1. Implement your objective function as simple as possible. You do not need to consider the computational efficiency at all. *"Done is better than perfect."*
      * Test with the toy example and make sure that your objective function returns zero for the ground-truth parameter.
   2. If your simple objective function above is unacceptably slow, try the followings in this order.
      1. Ask yourself again before trying to make it faster.  Is it really unacceptable?  If your calibration can finish in an hour and you do not do it so often, it might be OK for example. *"Premature optimization is the root of all evil."* (D. Knuth).
      2. Make sure that the calibration runs successfully anyway.  In what follows, double-check that the calibration results do not change before and after the code optimization.
      3. Vectorize the computation with `numpy`, i.e., no for-loops in the objective function.
         * or use [`numba`](https://numba.pydata.org/) (e.g. `@numba.jit`)
      4. If the system is sparse, use `jac_sparsity` option. It makes `scipy.optimize.least_squares` much faster even without analitical Jacobian.
      5. Implement the analytical Jacobian. You may want to use [maxima](http://wxmaxima-developers.github.io/wxmaxima/) to automate the calculation, or you may use [JAX](https://github.com/google/jax) or other autodiff solutions for this.
      6. Reimplement in C++ with [ceres-solver](http://ceres-solver.org/), [g2o](https://github.com/RainerKuemmerle/g2o), or [sba](http://users.ics.forth.gr/~lourakis/sba/) if the computation speed is really important.  You can also consider using PyTorch/Tensorflow for GPU-acceleration and autodiff by [Theseus](https://github.com/facebookresearch/theseus) or similar libraries.



[def]: https://colab.research.google.com/assets/colab-badge.svg
