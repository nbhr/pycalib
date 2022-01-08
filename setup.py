# Author: Shohei Nobuhara <nob@ieee.org>
# Copyright (c) 2018-2022 Shohei Nobuhara
# License: Apache Software License

from setuptools import setup
import pycalib

DESCRIPTION = "PyCalib: Simple Camera Calibration in Python for Beginners"
NAME = 'pycalib-simple'
AUTHOR = 'Shohei Nobuhara'
AUTHOR_EMAIL = 'nob@ieee.org'
URL = 'https://github.com/nbhr/pycalib'
LICENSE = 'Apache Software License'
DOWNLOAD_URL = 'https://github.com/nbhr/pycalib'
VERSION = pycalib.__version__
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'matplotlib>=3.5.1',
    'numpy>=1.22.0',
    'opencv_contrib_python>=4.5.5.62',
    'scikit_image>=0.19.1',
    'scipy>=1.7.3',
]

EXTRAS_REQUIRE = {
}

PACKAGES = [
    'pycalib'
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Image Processing',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Framework :: Matplotlib',
    'Framework :: Jupyter',
]

with open('README.md', 'r') as fp:
    readme = fp.read()
with open('CONTACT.txt', 'r') as fp:
    contacts = fp.read()
long_description = readme + '\n\n' + contacts

setup(name=NAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR,
      maintainer_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      long_description_content_type="text/markdown",
      long_description=long_description,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      packages=PACKAGES,
      classifiers=CLASSIFIERS
    )
