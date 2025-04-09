# EIDER: Emission Integral and Differential Emission Reconstruction

EIDER is a Python package for analyzing stellar emission lines and reconstructing the Differential Emission Measure (DEM) of stars. It builds on established astrophysics methods to provide a robust framework for understanding stellar atmospheres through spectral emission analysis.

## Overview

EIDER calculates Differential Emission Measure (DEM) distributions from observed emission line fluxes, leveraging MCMC sampling to fit DEM models to spectral data. This allows for the reconstruction of the temperature distribution of emitting plasma in stellar atmospheres, which is essential for understanding stellar coronal heating and activity.

The package can:
- Process stellar emission line data from UV and X-ray spectra
- Generate spectral contribution function matrices (G(T) matrices) using CHIANTI atomic database
- Fit DEM models using Markov Chain Monte Carlo (MCMC) techniques
- Synthesize spectra from DEM models
- Visualize DEM distributions and spectral features

## Installation

### Prerequisites

- Python 3.7+
- CHIANTI atomic database (v10.0+)
- Astropy, NumPy, SciPy, Matplotlib
- emcee (for MCMC sampling)
- TinyGP (for Gaussian process-based DEM models)
- JAX (for optimized numerical operations)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bellalongo/eider.git
cd eider
```

2. Set up CHIANTI environment variable:
```bash
export XUVTOP=/path/to/CHIANTI_database
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Core Components

### GTMatrix

The `GTMatrix` class handles the calculation of spectral contribution functions (G(T) matrices), which map plasma temperature to spectral emission. These matrices are essential for interpreting emission line fluxes in terms of DEM distributions.

```python
from eider.gtmatrix import GTMatrix

# Initialize the GTMatrix
gtmatrix = GTMatrix(star_config, gtmat_config, path_config)
gtmatrix.initialize()
gtmatrix.load_line_data()
gtmatrix.generate_all_matrices()
```

### DEM

The `DEM` class manages the Differential Emission Measure reconstruction using MCMC sampling to fit models to emission line data.

```python
from eider.dem import DEM

# Create DEM object and run MCMC
dem = DEM(gtmatrix, dem_config, star_config)
dem.run_mcmc()
dem.plot_dem()
```

### Spectrum

The `Spectrum` class synthesizes spectra from DEM models, allowing for comparison with observations and prediction of unmeasured spectral features.

```python
from eider.spectrum import Spectrum

# Generate synthetic spectrum
spectrum = Spectrum(dem)
spectrum.generate_spectrum()
spectrum.plot_spectrum()
```

## Configuration

EIDER uses dictionary-based configuration for flexibility. Example configuration:

```python
STAR_CONFIG = {
    'star_name': 'Tau-Ceti',
    'star_radius': 0.793 * u.Rsun,
    'star_distance': 3.65 * u.pc,
    'abundance': -0.5,
}

GTMAT_CONFIG = {
    'min_wavelength': 1,
    'max_wavelength': 1500,
    'rconst': 100,
    'min_templog': 4,
    'max_templog': 8,
    'npoints': 200,
    'pressure_list': [1e17]
}

DEM_CONFIG = {
    'n_walkers': 100,
    'burn_in_steps': 500,
    'production_steps': 1000,
    'thread_num': 4,
    'progress_interval': 100,
    'init_chebyshev': [22.0, -2.5, -0.8, -1.0, -0.5, -0.2, -0.1]
}
```

## Example Usage

```python
from eider.gtmatrix import GTMatrix
from eider.dem import DEM
from eider.spectrum import Spectrum
from eider.config import STAR_CONFIG, GTMAT_CONFIG, DEM_CONFIG, PATH_CONFIG
from eider.utils import create_directories, check_environment

# Create output directories
create_directories(PATH_CONFIG)

# Check CHIANTI environment
check_environment(PATH_CONFIG)

# Generate G(T) matrix
gtmatrix = GTMatrix(STAR_CONFIG, GTMAT_CONFIG, PATH_CONFIG)
gtmatrix.initialize()
gtmatrix.load_line_data()
gtmatrix.generate_all_matrices()

# Run DEM analysis
dem = DEM(gtmatrix, DEM_CONFIG, STAR_CONFIG)
dem.run_mcmc()
dem.plot_dem()

# Generate synthetic spectrum
spectrum = Spectrum(dem)
spectrum.generate_spectrum()
spectrum.plot_spectrum()
```

## Improvements from Previous Version

The current version of EIDER represents a significant improvement over the previous implementation:

1. **Modular Architecture**: Code is now organized into clear, focused classes with well-defined responsibilities
2. **Improved Configuration**: Flexible dictionary-based configuration system
3. **Enhanced Documentation**: Better docstrings and inline comments
4. **Memory Efficiency**: Optimized handling of large matrices and data structures
5. **Parallel Processing**: Better utilization of multiprocessing for MCMC
6. **Error Handling**: More robust error handling and user feedback
7. **Visualization**: Enhanced plotting capabilities for DEMs and spectra

## Future Work

- Add support for more DEM parametrization methods
- Implement additional spectral synthesis capabilities
- Expand compatibility with different spectral data formats
- Create interactive visualization tools
- Optimize performance for very large datasets

## Acknowledgments

This project builds upon the work of many researchers in stellar physics and atomic spectroscopy. Special thanks to:
- The CHIANTI atomic database team
- Contributors to the emcee MCMC library
- The astropy community for essential astronomical tools

## License

[MIT License](LICENSE)
