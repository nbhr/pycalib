# Simple Camera Calibration in Python for Beginners

## How to use

1. **Local:** You can clone/download this repository in to your local PC, and open `./ipynb/*.ipynb` files by your local Jupyter.
   * Required packages: `cv2`, `pytorch`
2. **Azure Notebooks:** You can use Microsoft Azure Notebooks to clone the repo and run `.ipynb` files online.
   * Open this repo in [![Azure Notebooks](https://notebooks.azure.com/launch.svg)](https://notebooks.azure.com/nbhr/projects/pycalib), and click `Clone` to start with your own copy.
3. **Colaboratory:** You can open each Jupyter notebook directly in Google Colaboratory by clicking the ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) buttons below.
   * *Warning:* Most of them do not run properly as-is, since colab does not clone images used in the Jupyter notebooks. Please upload required files manually.

## Single camera

1. [Intrinsic parameters from chessboard images](./ipynb/incalib.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/incalib.ipynb)
2. [Extrinsic parameters w.r.t. a chassboard](./ipynb/excalib-chess.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib-chess.ipynb)
3. [Distortion](./ipynb/distortion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/distortion.ipynb)

## Multiple cameras

4. [2-view extrinsic calibration from 2D-2D correspondences](./ipynb/excalib-2d.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/excalib-2d.ipynb)
5. [N-view registration](./ipynb/ncam_registration.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_registration.ipynb)
6. [N-view bundle adjustment](./ipynb/ncam_ba.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nbhr/pycalib/blob/master/ipynb/ncam_ba.ipynb)
