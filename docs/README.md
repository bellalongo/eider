# EIDER Documentation

This directory contains documentation for running and using the EIDER package.

## Getting Started

### Prerequisites

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

### Installation

Clone the repository:
```bash
git clone https://github.com/bellalongo/eider.git
cd eider
```

The code is not yet packaged for pip installation. You can run the scripts directly from the main directory or add the eider directory to your Python path:

```bash
# Add to Python path for the current session
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or in your script
import sys
sys.path.append('/path/to/eider')
```

## Usage Guide

EIDER can be used to analyze both emission line data and full spectral data. The workflow is controlled through two primary files:

1. First, configure your analysis by editing `config.py` with your parameters
2. Then run the analysis by executing `testing.py`

### Configuration (config.py)

Before running an analysis, you need to set up your configuration file (`config.py`) with the appropriate parameters:

```python
# Star parameters
STAR_NAME_ROOT = 'my_star'  # Root name for output files
STAR_TITLE = 'My Star'      # Title for plots
STAR_ABUNDANCE = 0.0        # Stellar abundance relative to solar
STAR_RADIUS = 0.75 * u.Rsun # Stellar radius
STAR_DISTANCE = 10.0 * u.pc # Distance to star

# Analysis parameters
PRESSURE_LIST = [1e17]     # Pressure values to test (dyne/cm²)
INIT_POSITION = [22.0, -3.0, -0.5, -1.0, -0.9, -0.6, -0.1]  # Initial guess for MCMC

# Input files
LINE_TABLE_FILE = 'my_star_linetable.ascii'  # For emission line analysis
SPECTRUM_DATA_FILE = None   # For spectral analysis (set to None if using only lines)
```

### Running the Analysis

Once you've configured `config.py`, simply run:

```bash
python testing.py
```

This will execute the full analysis pipeline:
1. Generate contribution functions
2. Process input data
3. Run MCMC fitting
4. Create output plots and files

### Output Files

EIDER generates several output files:

- `dem_[star_name].pdf`: DEM reconstruction plot
- `corner_[star_name].pdf`: Corner plot showing MCMC parameter distributions
- `compare_ion_[star_name].pdf`: Comparison between observed and modeled ion fluxes
- `spectrum_[star_name].fits`: Modeled spectrum based on the DEM
- `spectrum_[star_name].pdf`: Plot of the modeled spectrum

## Advanced Usage

### Customizing DEM Parameters

You can modify the DEM parameterization in `fitting.py`:

- For Chebyshev polynomial parameterization, adjust the degree of the polynomial (number of coefficients).
- For Gaussian process parameterization, adjust the kernel parameters.

### Pressure Modes

EIDER can run with different pressure assumptions:

- Constant pressure: Set a single pressure value in `press_list`
- Multi-pressure: Provide multiple pressure values to test different scenarios

### Abundance Modifications

To modify elemental abundances:

1. Set the `abundance` parameter in `run_single_star.py`
2. Choose the appropriate abundance file in `gofnt_routines.py`

## Troubleshooting

### Common Issues

1. **Missing atomic data**: Ensure ChiantiPy is properly configured with all required atomic databases.

2. **Memory issues**: G(n,T) calculations can be memory-intensive. Use the `get_gofnt_matrix_low_ram` function for large datasets.

3. **MCMC convergence issues**: Try adjusting the number of walkers, burn-in steps, or initial parameter guesses.

## Example Workflow

Here's an example workflow for analyzing a stellar spectrum:

1. Edit `config.py` to include your star's parameters:
   ```python
   # Star parameters
   STAR_NAME_ROOT = 'my_star'
   STAR_TITLE = 'My Star'
   STAR_ABUNDANCE = 0.0        # Solar abundance
   STAR_RADIUS = 1.0 * u.Rsun
   STAR_DISTANCE = 10.0 * u.pc
   
   # Input files - for emission line analysis
   LINE_TABLE_FILE = 'my_star_linetable.ascii'
   SPECTRUM_DATA_FILE = None
   ```

2. Prepare your emission line data in the required format

3. Run the analysis:
   ```bash
   python testing.py
   ```

4. Examine the output plots and FITS files
