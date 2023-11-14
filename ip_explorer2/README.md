# Overview

This package is intended to be used as a toolbox to aid in the development,
analysis, and benchmarking of interatomic potentials (IPs).

The core goals of this package are:
* to standardize boilerplate code for data loading, parallelization, and logging
* to create a generalizable framework for enabling easy extension to new models
* to accelerate the design and deployment of new tools for analyzing IPs

This package was developed as part of the work done in:
* J. Vita and D. Schwalbe-Koda, "[Data efficiency and extrapolation trends in neural network interatomic potentials](https://doi.org/10.1088/2632-2153/acf115)", _2023 Mach. Learn.: Sci.  Technol_.

## Installation
```
git clone https://github.com/jvita/ip_explorer.git
cd ip_explorer
pip install -e .
```

# Available tools
* simple energy/force predictions
* 1D and 2D loss landscape generation
* computing local environment descriptors
* (under development) potential energy surface visualization with [SHEAP](https://bitbucket.org/bshires/sheap/src/master/)
* (under development) [information imbalance](https://academic.oup.com/pnasnexus/article/1/2/pgac039/6568571) using [DADApy](https://github.com/sissa-data-science/DADApy) 

# Supported models
* [MACE](https://github.com/ACEsuit/mace)
* [NequIP](https://github.com/mir-group/nequip)
* [SchNet/PaiNN](https://github.com/atomistic-machine-learning/schnetpack)

# Core dependencies
* [ASE](https://wiki.fysik.dtu.dk/ase/)
* [PytorchLightning](https://lightning.ai/docs/pytorch/stable/)
* [loss-landscapes](https://github.com/marcellodebernardi/loss-landscapes)
  * This package has been patched to allow for more efficient evaluation of the landscapes. The patched version of the code is included in [patches/loss_landscapes](https://github.com/jvita/ip_explorer/tree/main/patches/loss_landscapes)

# Code structure

## Folders
* `ip_explorer`
    * `datamodules/`: for loading and pre-processing datasets
    * `models/`: for loading models, performing parallel inference, and aggregating results
    * `landscape/`: for computing loss landscapes
    * `representations/`: for computing local environment descriptors
    * `pes/`: for generating inputs to the external SHEAP tool
    * `information_imbalance`: for computing information imbalances

## Contributing new models
This package works by wrapping models to provide them with a
PytorchLightning-compliant interface, which is then used for handling parallel
evaluation. The model wrapper also specifies how data and computed values should
be aggregated in order to properly compute metrics used by the core tools.
Typically, these include things like error values, latent representations, or
other properties of interest.

A user interested in extending `ip_explorer` to support new models should look
at the `PLModelWrapper` class in `models/base.py`, from which all models should
sub-class. In particular, the `load_model()`, `compute_loss()`, and
`aggregate_loss()` functions will likely need to be customized for the given
model.

## Contributing new tools
Adding a new tool to `ip_explorer` usually only involves writing a script to
perform the desired functionality, leveraging the parallelization provided by
the `PLModelWrapper` class and the supported properties that can be
computed/aggregated using the `compute_*()` and `aggregate_*()` functions. Any
model that implements the necessary compute/aggregate functions will then be
able to use the new tool.

# Citing
If you use `ip_explorer` in a publication, please cite the following paper:

```
@misc{Vita2023,
  title = {Data efficiency and extrapolation trends in neural network interatomic potentials},
  author = {Vita,  Joshua A. and Schwalbe-Koda,  Daniel},
  publisher = {arXiv},
  year = {2023},
  doi = {10.48550/ARXIV.2302.05823},
  url = {https://arxiv.org/abs/2302.05823},
}
```

# Authors
* Josh Vita (vita1@llnl.gov)
* Daniel Schwalbe-Koda (schwalbekoda1@llnl.gov)

## Release

The data is distributed under the following license: CC BY 4.0

LLNL-CODE-852928
