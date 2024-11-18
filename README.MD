# Pykatsevich

## Introduction

A Python package with the implementation of Katsevich algorithm for helical CT.
A good algorithm description is given in [F. Noo et al, Exact helical reconstruction using native cone-beam geometries](https://iopscience.iop.org/article/10.1088/0031-9155/48/23/001).
Filtering steps run on CPU.
Backprojection runs on a GPU, thanks to the `BP3D_CUDA` algorithm from the [ASTRA toolbox](https://github.com/astra-toolbox/astra-toolbox).

## Dependencies
`astra` >= 2, `numpy`, `cupy`, `scipy`.

## Installation

Clone the repository, then run:
```
cd pykatsevich
make install
```
This will run `pip install . --verbose` and clean up after the installation is done.
To uninstall the package, run `make uninstall`.
## Usage
A typical pipeline includes algorithm initialization (module `initialize`), preparation of projection and volume geometry (module `geometry`), filtering (module `filtering`), and reconstruction (module `reconstruct`).
For a quick example, run `python tests/t03.py`.

## Authors and acknowledgment
Pavel Paramonov and Femke Danckaers.

## License
GPL v3
