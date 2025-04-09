from fitting import MCMCFitter, MCMCConfig
from probability_fxns import ln_likelihood_dem
import numpy as np
from astropy import units as u
from typing import Tuple, Any, Dict, Optional
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval
from scipy.integrate import cumulative_trapezoid
import roman
import os
from os.path import exists

class DEM:
    """

    """
    def __init__(self, 
                 gtmatrix, 
                 dem_config: Dict,
                 star_config):
        """
        Initialize the DEM object.
        
        Args:
            gtmatrix: The GTMatrix object containing contribution functions
            dem_config: Configuration dictionary for DEM analysis
            star_config: Star parameters (optional if provided in gtmatrix)
        """
        self.gtmatrix = gtmatrix
        self.dem_config = dem_config
        self.star_config = star_config
            
        # Load configuration data
        self._load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs('mcmc', exist_ok=True)
        
        # MCMC result files
        self.sample_dir = f'mcmc/samples_{self.star_name}'
        self.lnprob_dir = f'mcmc/lnprob_{self.star_name}'
        
        # Initialize temperature grid
        self.temp = np.logspace(self.gtmatrix.min_templog, self.gtmatrix.max_templog, self.gtmatrix.npoints)
        self.log_temp = np.log10(self.temp)
        
        # Calculate flux weighting based on stellar geometry
        self.flux_weighting = ((np.pi * u.sr * (self.radius**2.0) * 
                               (1.0 / (self.distance**2.0))).to(u.sr)).value
        
        # Initialize samples and lnprob as None (will be set by run_mcmc)
        self.samples = None
        self.lnprob = None
        
        print(f"DEM initialized for {self.star_name}")
        print(f"Using {len(self.gtmatrix.ion_fluxes)} emission lines")

    # ------------------------------
    # Private Helper Methods
    # ------------------------------
    def _load_data(self):
        """

        """
        # Extract star parameters
        self.star_name = self.star_config['star_name']
        self.radius = self.star_config['star_radius']
        self.distance = self.star_config['star_distance']
        
        # Extract MCMC parameters
        self.n_walkers = self.dem_config['n_walkers']
        self.burn_in_steps = self.dem_config['burn_in_steps']
        self.production_steps = self.dem_config['production_steps']
        self.thread_num = self.dem_config['thread_num']
        self.progress_interval = self.dem_config['progress_interval']
        
        # Extract initial Chebyshev coefficients
        self.init_chebyshev = self.dem_config['init_chebyshev']

    def _check_existing_results(self) -> bool:
        """

        """
        return exists(self.sample_dir) and exists(self.lnprob_dir)

    
    # ------------------------------
    # Public Methods
    # ------------------------------
    def load_results(self) -> bool:
        """

        """
        if self._check_existing_results():
            print(f"Loading existing MCMC results from {self.sample_dir}")
            self.samples = np.load(self.sample_dir)
            self.lnprob = np.load(self.lnprob_dir)
            return True
        return False

    def run_mcmc(self) -> Tuple[np.ndarray, np.ndarray, Any]:
        """

        """
        # Check for existing results first
        if self.load_results():
            return self.samples, self.lnprob, None
        
        print(f"Running MCMC for {self.star_name}")
        
        # Get flux and error data
        flux = self.gtmatrix.ion_fluxes
        err = self.gtmatrix.ion_errs

        # Get the emission line indices by calling the method
        indices = self.gtmatrix.get_emission_line_indices()

        # Make sure indices are within bounds
        valid_indices = [idx for idx in indices if idx < self.gtmatrix.gtmat.shape[0]]

        # Get the subset of the G(T) matrix using valid indices
        gtmat_subset = self.gtmatrix.gtmat[valid_indices]
        
        # Create MCMC configuration
        config = MCMCConfig(
            n_walkers=self.n_walkers,
            burn_in_steps=self.burn_in_steps,
            production_steps=self.production_steps,
            thread_num=self.thread_num,
            progress_interval=self.progress_interval
        )
        
        # Initialize fitter
        fitter = MCMCFitter(config)
        
        
        # Package likelihood arguments with the subset of G(T) matrix
        likelihood_args = [
            flux,
            err, 
            self.log_temp,
            self.temp,
            gtmat_subset,  
            self.flux_weighting
        ]

        # Run MCMC
        print(f"Starting MCMC with {self.n_walkers} walkers, {self.burn_in_steps} burn-in steps, and {self.production_steps} production steps")
        samples, lnprob, sampler = fitter.fit(
            init_pos=self.init_chebyshev,
            likelihood_func=ln_likelihood_dem,
            likelihood_args=likelihood_args
        )

        # Save results
        print(f"MCMC complete. Saving results to {self.sample_dir}")
        np.save(self.sample_dir, samples)
        np.save(self.lnprob_dir, lnprob)
        
        # Update instance variables
        self.samples = samples
        self.lnprob = lnprob

        return samples, lnprob, sampler
    
    def plot_dem(self, 
                sample_num: int = 500, 
                main_color: str = 'b', 
                sample_color: str = 'cornflowerblue', 
                alpha: float = 0.1,
                low_y: float = 19.0, 
                high_y: float = 26.0,
                save_path: str = None) -> plt.Figure:
        """

        """
        # Check if MCMC has been run
        if self.samples is None or self.lnprob is None:
            print("No MCMC results available. Run run_mcmc() first.")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Prepare temperature grid for Chebyshev evaluation
        shift_log_temp = self.log_temp - np.mean(self.log_temp)
        range_temp = (np.max(shift_log_temp) - np.min(shift_log_temp))
        shift_log_temp /= (0.5 * range_temp)
        
        # Get best-fit model (highest probability sample)
        best_idx = np.argmax(self.lnprob)
        psi_model = 10.0**chebval(shift_log_temp, self.samples[best_idx][:-1])
        
        # Calculate flux constraints for visualization
        flux_arr = self.gtmatrix.ion_fluxes
        gofnt_matrix = self.gtmatrix.gtmat
        psi_ys = flux_arr / (self.flux_weighting * np.trapz(gofnt_matrix, self.temp))
        
        # Find temperature range for each constraint
        temp_lows = 1e4 * np.ones_like(psi_ys) # FIX ?
        temp_upps = 1e8 * np.ones_like(temp_lows)
        
        for i in range(len(flux_arr)):
            gofnt_cumtrapz = cumulative_trapezoid(gofnt_matrix[i], self.temp)
            low_index = np.argmin(
                np.abs(gofnt_cumtrapz - (0.16 * gofnt_cumtrapz[-1])))
            upp_index = np.argmin(
                np.abs(gofnt_cumtrapz - (0.84 * gofnt_cumtrapz[-1])))
            temp_lows[i] = self.temp[low_index + 1]
            temp_upps[i] = self.temp[upp_index + 1]
        
        # Plot random samples from MCMC
        total_samples = np.random.choice(len(self.samples), sample_num)
        for i in range(sample_num):
            s = self.samples[total_samples[i]]
            temp_psi = 10.0**chebval(shift_log_temp, s[:-1])
            if i == 0:
                plt.loglog(self.temp, temp_psi,
                        color=sample_color, alpha=alpha, label='MCMC Samples')
            else:
                plt.loglog(self.temp, temp_psi, color=sample_color, alpha=alpha)
        
        # Plot best-fit model
        plt.loglog(self.temp, psi_model, color=main_color, label='Best-fit DEM model')
        
        # Plot flux constraints
        plt.hlines(psi_ys, temp_lows, temp_upps, label='Flux Constraints',
                colors='k', zorder=100)
        
        # Add ion labels if available
        if hasattr(self.gtmatrix, 'ion_list') and self.gtmatrix.ion_list is not None:
            ion_names = self.gtmatrix.ion_list
            ion_gofnts = gofnt_matrix
            ion_fluxes = flux_arr
            
            dem_xs = np.array([self.temp[np.argmax(ion_gofnts[i])] for i in range(len(ion_names))])
            dem_ys = ion_fluxes
            
            # Use GTMatrix.calculate_integral instead of do_gofnt_matrix_integral
            ones_array = np.ones_like(self.temp)
            dem_ys /= self.gtmatrix.calculate_integral(
                ion_gofnts, 
                self.temp, 
                ones_array, 
                self.flux_weighting
            )
            
            for i in range(len(ion_names)):
                ion_name = ion_names[i].split('_')
                new_name = ion_name[0].capitalize() + ' '
                new_name += roman.toRoman(int(ion_name[1]))
                plt.text(dem_xs[i], dem_ys[i], new_name)
        
        # Set plot limits and labels
        plt.ylim(10.0**low_y, 10.0**high_y)
        plt.xlabel('Temperature [K]')
        y_label = r'$\Psi(T) = N_e N_{\mathrm{H}} \frac{ds}{dT}$ '
        y_label += r'[cm$^{-5}$ K$^{-1}$]'
        plt.ylabel(y_label)
        plt.title(f'{self.star_name} DEM')
        plt.legend()
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"DEM plot saved to {save_path}")
        
        return plt.gcf()