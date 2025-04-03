from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import os
from typing import Optional, Tuple, Dict
from pathlib import Path

class Spectrum:
    """

    """
    
    def __init__(self, dem):
        """
        Initialize the Spectrum object with a DEM model.
        
        Args:
            dem: DEM object containing the differential emission measure model
        """
        self.dem = dem
        self.spectrum_table = None
        self.wave_arr = None
        self.bin_arr = None
        self.flux = None
        self.err = None
        self.best_spectra = None
        
        # Set up paths
        self.spectrum_dir = 'spectra'
        os.makedirs(self.spectrum_dir, exist_ok=True)
        self.spectrum_path = f'{self.spectrum_dir}/spectrum_{self.dem.star_name}.fits'
    
    def generate_spectrum(self, sample_num: int = 1000) -> Table:
        """

        """
        # Check if MCMC samples are available
        if self.dem.samples is None:
            print("No MCMC samples available. Run dem.run_mcmc() first.")
            return None
            
        # Check if spectrum already exists
        if os.path.isfile(self.spectrum_path):
            print(f"Loading existing spectrum from {self.spectrum_path}")
            self.spectrum_table = Table.read(self.spectrum_path)
            return self.spectrum_table
            
        # Get wavelength grid from GTMatrix
        self.wave_arr = self.dem.gtmatrix.wave_arr
        self.bin_arr = self.dem.gtmatrix.bin_arr
        
        # Get full GTMatrix
        gofnt_matrix = self.dem.gtmatrix.gtmat
        
        # Generate spectrum from samples
        print(f"Generating spectrum for {self.dem.star_name}...")
        self.spectrum_table, self.best_spectra = self._generate_spectrum_from_samples(
            gofnt_matrix,
            self.dem.samples,
            self.dem.lnprob,
            self.dem.flux_weighting,
            self.wave_arr,
            self.bin_arr,
            sample_num
        )
        
        return self.spectrum_table
    
    def plot_spectrum(self, 
                     alpha: float = 0.3, 
                     color: str = 'b',
                     save_path: str = None) -> plt.Figure:
        """

        """
        # Check if spectrum has been generated
        if self.spectrum_table is None:
            print("No spectrum available. Run generate_spectrum() first.")
            if os.path.isfile(self.spectrum_path):
                self.spectrum_table = Table.read(self.spectrum_path)
            else:
                return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Get data from table
        wave = self.spectrum_table['Wavelength']
        flux = self.spectrum_table['Flux_density']
        lower_err = self.spectrum_table['Lower_Error_16']
        upper_err = self.spectrum_table['Upper_Error_84']
        
        # Plot the spectrum
        plt.semilogy(wave, flux, drawstyle='steps-mid', color=color)
        plt.fill_between(wave, flux - lower_err, flux + upper_err, 
                         color=color, alpha=alpha, step='mid')
        
        # Add labels
        plt.title(f'{self.dem.star_name} Synthetic Spectrum')
        plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
        plt.ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Spectrum plot saved to {save_path}")
            
        return plt.gcf()
    
    def _generate_spectrum_from_samples(self, 
                                       gofnt_matrix,
                                       samples, 
                                       lnprob,
                                       flux_weighting,
                                       wave_arr,
                                       bin_arr,
                                       sample_num: int = 1000) -> Tuple[Table, np.ndarray]:
        """
        
        """
        save_name = self.spectrum_path.replace('.fits', '')
        
        # Set up temperature grid
        temp = self.dem.temp
        log_temp = self.dem.log_temp
        shift_log_temp = log_temp - np.mean(log_temp)
        range_temp = (np.max(shift_log_temp) - np.min(shift_log_temp))
        shift_log_temp /= (0.5 * range_temp)

        # Calculate best model spectrum
        best_idx = np.argmax(lnprob)
        best_psi = 10.0**np.polynomial.chebyshev.chebval(shift_log_temp, samples[best_idx][:-1])
        
        best_spectra2 = self.dem.gtmatrix.calculate_integral(
            gofnt_matrix, 
            temp, 
            best_psi, 
            flux_weighting
        )
        best_spectra2 /= bin_arr

        # Generate spectra from random samples
        all_indices = np.arange(0, len(samples))
        rand_indices = np.random.choice(all_indices, sample_num)
        spec_len = len(wave_arr)
        all_spectra = np.zeros((sample_num, spec_len))
        all_psi = np.zeros((sample_num, len(temp)))

        for i in range(sample_num):
            sample = samples[rand_indices[i]]
            temp_psi = 10.0**np.polynomial.chebyshev.chebval(shift_log_temp, sample[:-1])
            temp_spectra = self.dem.gtmatrix.calculate_integral(
                gofnt_matrix, 
                temp, 
                temp_psi, 
                flux_weighting
            )
            temp_err = temp_spectra * (10.0**(sample[-1]))
            all_spectra[i, :] = np.random.normal(loc=temp_spectra, scale=temp_err)
            all_spectra[i, :] /= bin_arr
            all_psi[i, :] = temp_psi

        # Calculate median and percentiles
        wave_unit = u.Angstrom
        flux_unit = u.erg / (u.s * u.cm**2)
        best_spectra = np.median(all_spectra, axis=0)

        med_psi = np.median(all_psi, axis=0)
        upp_psi = np.percentile(all_psi, 84, axis=0)
        low_psi = np.percentile(all_psi, 16, axis=0)

        upper_diff_var = (np.percentile(all_spectra, 84, axis=0) - best_spectra)**2
        lower_diff_var = (best_spectra - np.percentile(all_spectra, 16, axis=0))**2

        upper_err = np.sqrt(upper_diff_var)
        lower_err = np.sqrt(lower_diff_var)

        # Create and save table
        spectrum_table = Table([wave_arr * wave_unit,
                               best_spectra * flux_unit,
                               lower_err * flux_unit,
                               upper_err * flux_unit,
                               best_spectra2 * flux_unit],
                              names=('Wavelength', 'Flux_density',
                                    'Lower_Error_16', 'Upper_Error_84',
                                    'Flux_density_ln_lmax'))
        spectrum_table.write(f'{save_name}.fits', format='fits', overwrite=True)
        np.save(f'{save_name}_dems.npy', [low_psi, med_psi, upp_psi])
        
        return spectrum_table, best_spectra