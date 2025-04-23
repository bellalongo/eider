# EIDER

**E**mission **I**ntegral **D**ifferential **E**mission measure **R**econstruction

## Overview

EIDER is a Python package for reconstructing differential emission measure (DEM) distributions from spectroscopic data of astrophysical plasmas. It uses a forward modeling approach to infer the underlying temperature distribution of the plasma based on observed emission lines or spectra.

The code is particularly useful for analyzing stellar coronae, where plasma exists at a range of temperatures simultaneously. EIDER provides tools to:

1. Generate contribution functions (G(n,T)) for various ions and transitions
2. Fit emission measure distributions using MCMC techniques
3. Visualize results through temperature-emission measure plots
4. Compare model predictions with observed spectra or emission line fluxes

## Features

- Integration with ChiantiPy atomic database for atomic physics calculations
- MCMC fitting of differential emission measure distributions
- Visualization tools for DEM analysis
- Support for both emission line and spectral data input
- Flexible parameterization of DEM with Chebyshev polynomials or Gaussian processes

## Installation

```bash
git clone https://github.com/bellalongo/eider.git
cd eider
```

The code is not yet packaged for pip installation. Simply add the directory to your Python path or run scripts from the main directory.

## Dependencies
Before using EIDER, ensure you have the following dependencies installed:

[![astropy](https://img.shields.io/badge/astropy-red?style=for-the-badge)](https://astropy.org/)
[![ChiantiPy](https://img.shields.io/badge/ChiantiPy-orange?style=for-the-badge)](https://chianti-atomic.github.io/)
[![corner](https://img.shields.io/badge/corner-yellow?style=for-the-badge)](https://corner.readthedocs.io/)
[![emcee](https://img.shields.io/badge/emcee-yellowgreen?style=for-the-badge)](https://emcee.readthedocs.io/)
[![jax](https://img.shields.io/badge/jax-green?style=for-the-badge)](https://jax.readthedocs.io/)
[![matplotlib](https://img.shields.io/badge/matplotlib-greenblue?style=for-the-badge)](https://matplotlib.org/stable/index.html)
[![numpy](https://img.shields.io/badge/numpy-blue?style=for-the-badge)](https://numpy.org/doc/)
[![pandas](https://img.shields.io/badge/pandas-darkblue?style=for-the-badge)](https://pandas.pydata.org/docs/)
[![Python](https://img.shields.io/badge/Python-indigo?style=for-the-badge)](https://www.python.org/)
[![scipy](https://img.shields.io/badge/scipy-violet?style=for-the-badge)](https://scipy.org/)
[![seaborn](https://img.shields.io/badge/seaborn-darkviolet?style=for-the-badge)](https://seaborn.pydata.org/)
[![tinygp](https://img.shields.io/badge/tinygp-purple?style=for-the-badge)](https://tinygp.readthedocs.io/)

ChiantiPy requires additional setup to access atomic databases. Please refer to the [ChiantiPy documentation](https://chianti-atomic.github.io/) for installation instructions.

## Implementation Status
Comparative code is found in the `original` directory

### Completed
- [x] Basic framework for DEM reconstruction
- [x] Contribution function (G(n,T)) calculation
- [x] MCMC fitting infrastructure
- [x] Visualization tools for DEM and spectra
- [x] Integration with ChiantiPy atomic database
- [x] Flux-conserving resampling functions

### To Do
- [ ] MCMC fit for low-resolution data
- [ ] Fix spectrum code functionality 
- [ ] Fix errors in workflow for when wavelength arrays need to be resampled
- [ ] Implement Gaussian Process DEM parameterization properly
- [ ] Fix integration between emission line and spectral data workflows
- [ ] Implement proper ion selection for contribution functions
- [ ] Create plotting functions for direct comparison of observed vs. model spectra
- [ ] Fix error handling for ChiantiPy integration
- [ ] Implement proper file management for intermediate calculation results
- [ ] Fix normalization issues in DEM calculations
- [ ] Complete workflow example for new users
