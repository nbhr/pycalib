# Simple Camera Calibration in Python for Beginners

## How to use

1. **Local:** You can clone/download this repository in to your local PC, and open `./ipynb/*.ipynb` files by Jupyter.
   * Required packages: `cv2`, `pytorch`
2. **Azure Notebooks:** You can use Microsoft Azure Notebooks to clone the repo and run `.ipynb` files online.
   * Open this repo in [![Azure Notebooks](https://notebooks.azure.com/launch.svg)](https://notebooks.azure.com/nbhr/projects/pycalib), and click `Clone` to start with your own copy.
3. **Colaboratory:** You can open each Jupyter notebooks directly in Google Colaboratory by clicking the ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) buttons below.
   * Since colab does not clone images used in the Jupyter notebooks, some of them do not run properly as-is.  Please upload required files by hand.

## Lv1: OpenCV

* [Intrinsic parameters from chessboard images](./ipynb/incalib.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/incalib.ipynb)
* [Extrinsic parameters w.r.t. a chassboard](./ipynb/excalib-chess.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib-chess.ipynb)
* [Extrinsic parameters from 2D-2D correspondences](./ipynb/excalib-2d.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib-2d.ipynb)
* [Distortion](./ipynb/distortion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/distortion.ipynb)

## Lv2: Multi-camera

* [N-view registration](./ipynb/ncam_registration.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_registration.ipynb)
* [Bundle adjustment](./ipynb/ncam_ba.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_ba.ipynb)