from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.units import Quantity

@dataclass
class SpectrumConfig:
    """Configuration parameters for spectrum processing."""
    wave_min: float = 5.0
    wave_max: float = 150.0
    energy_step: float = 5e-3  # keV
    flux_threshold: float = 1.0  # SNR threshold
    skiprows: int = 3
    nrows: int = 1500

class SpectrumProcessor:
    def __init__(self, config: Optional[SpectrumConfig] = None):
        """Initialize the spectrum processor.
        
        Args:
            config: Configuration parameters for processing. If None, uses defaults.
        """
        self.config = config or SpectrumConfig()
        self._spec_df: Optional[pd.DataFrame] = None
        self._resid_df: Optional[pd.DataFrame] = None

        
    @property
    def column_names(self) -> Dict[str, list]:
        """Define column names for the data frames."""
        return {
            'spec': ['energy', 'energy_err', 'spec', 'spec_err', 
                    'bestfit_model', 'mod1', 'mod2', 'mod3', 'mod4'],
            'resid': ['energy', 'energy_err', 'resid', 'resid_err',
                     'no1', 'no2', 'no3', 'no4', 'no5']
        }
    

    def read_data(self, filepath: Path) -> None:
        """Read spectrum and residual data from file.
        
        Args:
            filepath: Path to the data file.
        """
        self._spec_df = pd.read_csv(
            filepath,
            skiprows=self.config.skiprows,
            nrows=self.config.nrows,
            header=None,
            delim_whitespace=True,
            names=self.column_names['spec']
        )
        
        self._resid_df = pd.read_csv(
            filepath,
            skiprows=self.config.skiprows,
            nrows=self.config.nrows,
            header=None,
            delim_whitespace=True,
            names=self.column_names['resid']
        )


    def plot_initial_data(self) -> None:
        """Plot the initial spectrum data for visualization."""
        if self._spec_df is None:
            raise ValueError("No data loaded. Call read_data() first.")
            
        plt.figure(figsize=(10, 6))
        plt.errorbar(self._spec_df['energy'], 
                    self._spec_df['spec'],
                    label='data',
                    drawstyle='steps-mid')
        plt.plot(self._spec_df['energy'],
                self._spec_df['bestfit_model'],
                label='best')
        plt.plot(self._spec_df['energy'],
                self._spec_df['mod1'],
                label='mod1')
        plt.plot(self._spec_df['energy'],
                self._spec_df['mod2'],
                label='mod2')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()
        plt.tight_layout()


    def _calculate_wavelengths(self, energy: Quantity) -> Tuple[Quantity, ...]:
        """Calculate wavelength arrays from energy values.
        
        Args:
            energy: Energy values with units.
            
        Returns:
            Tuple containing wavelength arrays (wave_lows, wave_upps, wave, bins).
        """
        energy_upps = energy + (self.config.energy_step * u.keV)
        energy_lows = energy - (self.config.energy_step * u.keV)
        
        wave_lows = energy_upps.to(u.AA, equivalencies=u.spectral())
        wave_upps = energy_lows.to(u.AA, equivalencies=u.spectral())
        wave = 0.5 * (wave_lows + wave_upps)
        bins = wave_upps - wave_lows
        
        return wave_lows, wave_upps, wave, bins
    

    def process_spectrum(self, star_name: str, freefree_correction: bool = False) -> Tuple[Quantity, ...]:
        """Process spectrum data and convert to wavelength space.
        
        Args:
            star_name: Name of the star being processed.
            freefree_correction: Whether to apply free-free correction.
            
        Returns:
            Tuple containing (wavelength, bins, flux, error) arrays with units.
        """
        if self._spec_df is None:
            raise ValueError("No data loaded. Call read_data() first.")

        # Convert energy to astropy quantity
        energy = np.array(self._spec_df['energy']) * u.keV
        
        # Calculate wavelength arrays
        _, _, wave, bins = self._calculate_wavelengths(energy)
        
        # Process flux
        flux_unit_old = 1.0 / (u.s * u.cm * u.cm * u.keV)
        if freefree_correction:
            flux_old = np.array(self._spec_df['spec'] - self._spec_df['mod2']) * flux_unit_old
        else:
            flux_old = np.array(self._spec_df['spec']) * flux_unit_old
            
        flux_old *= energy.to(u.erg)
        err_old = np.array(self._spec_df['spec_err']) * flux_unit_old * energy.to(u.erg)
        
        # Convert to new flux units
        flux_unit_new = flux_unit_old * u.erg * u.keV / u.AA
        flux = ((flux_old * 1e-2 * u.keV) / bins).to(flux_unit_new)
        err = ((err_old * 1e-2 * u.keV) / bins).to(flux_unit_new)
        
        # Apply quality masks
        mask = (
            (flux.value > 0.0) &
            (err.value > 0.0) &
            (flux > self.config.flux_threshold * err) &
            np.isfinite(flux.value) &
            np.isfinite(err.value) &
            (wave.value <= self.config.wave_max) &
            (wave.value >= self.config.wave_min)
        )
        
        return wave[mask], bins[mask], flux[mask], err[mask]


    def plot_processed_spectrum(self, wave: Quantity, flux: Quantity, 
                              err: Quantity) -> None:
        """Plot the processed spectrum data.
        
        Args:
            wave: Wavelength array with units
            flux: Flux array with units
            err: Error array with units
        """
        plt.figure(figsize=(10, 6))
        plt.errorbar(wave, flux, yerr=err, drawstyle='steps-mid')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel(f'Flux ({flux.unit})')
        plt.tight_layout()


    def save_spectrum_data(self, output_path: Path, wave: Quantity, 
                          bins: Quantity, flux: Quantity, err: Quantity) -> None:
        """Save the processed spectrum data.
        
        Args:
            output_path: Path to save the data
            wave: Wavelength array with units
            bins: Bin width array with units
            flux: Flux array with units
            err: Error array with units
        """
        np.save(
            output_path,
            [wave.value, bins.value, flux.value, err.value]
        )


    def process_spectrum_data(data_file: Path, star_name: str, 
                            output_dir: Path, config: Optional[SpectrumConfig] = None,
                            freefree_correction: bool = False) -> Path:
        """Process spectrum data for a given star.
        
        Args:
            data_file: Path to input data file
            star_name: Name of the star
            output_dir: Directory to save outputs
            config: Processing configuration
            freefree_correction: Whether to apply free-free correction
            
        Returns:
            Path to the saved spectrum data file
        """
        # Initialize processor
        processor = SpectrumProcessor(config)
        
        # Read and process data
        processor.read_data(data_file)
        processor.plot_initial_data()
        
        wave, bins, flux, err = processor.process_spectrum(
            star_name,
            freefree_correction
        )
        
        # Plot processed data
        processor.plot_processed_spectrum(wave, flux, err)
        
        # Save results
        output_path = output_dir / f"{star_name}_spectrum_data.npy"
        processor.save_spectrum_data(output_path, wave, bins, flux, err)
        
        return output_path