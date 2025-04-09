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
        self.sample_file = f'{self.sample_dir}.npy'
        self.lnprob_file = f'{self.lnprob_dir}.npy'
        
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
        return exists(self.sample_file) and exists(self.lnprob_file)

    
    # ------------------------------
    # Public Methods
    # ------------------------------
    def load_results(self) -> bool:
        """

        """
        if self._check_existing_results():
            print(f"Loading existing MCMC results from {self.sample_dir}")
            self.samples = np.load(self.sample_file)
            self.lnprob = np.load(self.lnprob_file)
            return True
        return False

    def run_mcmc(self) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Run MCMC sampling to fit DEM model to data or load existing results.
        
        Returns:
            Tuple containing samples, log probabilities, and sampler object
        """
        # Check for existing results first
        if not self.load_results():
            print(f"Running MCMC for {self.star_name}")
            
            # Get flux and error data
            flux = self.gtmatrix.ion_fluxes
            err = self.gtmatrix.ion_errs
            
            # Get indices of emission lines with valid measurements
            indices = self.gtmatrix.get_emission_line_indices()
            
            # Print diagnostics
            print(f"Found {len(indices)} emission lines with valid measurements")
            print(f"Flux values: {flux}")
            print(f"Error values: {err}")
            
            # Get the subset of the G(T) matrix
            gtmat_subset = np.array([self.gtmatrix.gtmat[idx] for idx in indices if idx < self.gtmatrix.gtmat.shape[0]])
            
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
            
            # Package likelihood arguments
            likelihood_args = [
                flux,
                err, 
                self.log_temp,
                self.temp,
                gtmat_subset,  
                self.flux_weighting
            ]

            # Run MCMC
            print(f"Starting MCMC with {self.n_walkers} walkers")
            samples, lnprob, sampler = fitter.fit(
                init_pos=self.init_chebyshev,
                likelihood_func=ln_likelihood_dem,
                likelihood_args=likelihood_args
            )

            # Save results
            np.save(self.sample_dir, samples)
            np.save(self.lnprob_dir, lnprob)
            
            # Update instance variables
            self.samples = samples
            self.lnprob = lnprob
            
            return samples, lnprob, sampler
        
        return self.samples, self.lnprob, None
    
    def plot_dem(self, 
                sample_num: int = 500, 
                main_color: str = 'b', 
                sample_color: str = 'cornflowerblue', 
                alpha: float = 0.1,
                low_y: float = 19.0, 
                high_y: float = 26.0) -> plt.Figure:
        """
        Plot the Differential Emission Measure (DEM) with samples from MCMC.
        
        Args:
            sample_num: Number of samples to plot
            main_color: Color for the best-fit model
            sample_color: Color for the MCMC samples
            alpha: Transparency for sample lines
            low_y: Lower y-axis limit in log10 space
            high_y: Upper y-axis limit in log10 space
            
        Returns:
            Matplotlib figure object
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
        
        # Get flux and G(T) data
        flux_arr = self.gtmatrix.ion_fluxes
        
        # IMPORTANT: Get G(T) matrix for emission lines only
        indices = self.gtmatrix.get_emission_line_indices()
        gofnt_matrix = np.array([self.gtmatrix.gtmat[idx] for idx in indices if idx < self.gtmatrix.gtmat.shape[0]])
        
        # Calculate psi_y values properly
        psi_ys = []
        temp_lows = []
        temp_upps = []
        
        for i in range(len(indices)):
            if i < len(gofnt_matrix):
                # Calculate integral of G(T) for this emission line
                integral = np.trapz(gofnt_matrix[i], self.temp)
                if integral > 0:
                    # This is the key calculation for flux constraints
                    psi_y = flux_arr[i] / (self.flux_weighting * integral)
                    psi_ys.append(psi_y)
                    
                    # Calculate temperature range for this line
                    gofnt_cum = np.cumsum(gofnt_matrix[i] * np.diff(self.temp)[0])
                    max_val = gofnt_cum[-1]
                    if max_val > 0:
                        low_idx = np.argmin(np.abs(gofnt_cum - (0.16 * max_val)))
                        upp_idx = np.argmin(np.abs(gofnt_cum - (0.84 * max_val)))
                        temp_lows.append(self.temp[max(0, low_idx)])
                        temp_upps.append(self.temp[min(len(self.temp)-1, upp_idx)])
                    else:
                        temp_lows.append(self.temp[0])
                        temp_upps.append(self.temp[-1])
        
        # Plot MCMC samples
        total_samples = np.random.choice(len(self.samples), sample_num)
        for i in range(sample_num):
            s = self.samples[total_samples[i]]
            temp_psi = 10.0**chebval(shift_log_temp, s[:-1])
            label = 'MCMC Samples' if i == 0 else None
            plt.loglog(self.temp, temp_psi, color=sample_color, alpha=alpha, label=label)
        
        # Plot best-fit model
        plt.loglog(self.temp, psi_model, color=main_color, label='Best-fit DEM model')
        
        # Plot flux constraints - make sure they're visible
        for i in range(len(psi_ys)):
            plt.hlines(psi_ys[i], temp_lows[i], temp_upps[i], 
                    colors='k', linewidth=2, zorder=100)
        
        # Add a collective label for flux constraints
        plt.plot([], [], 'k-', linewidth=2, label='Flux Constraints')
        
        # Add ion labels if available
        if hasattr(self.gtmatrix, 'ion_list') and self.gtmatrix.ion_list is not None:
            line_temps = [self.temp[np.argmax(gofnt_matrix[i])] for i in range(len(gofnt_matrix))]
            for i, name in enumerate(self.gtmatrix.ion_list):
                if i < len(line_temps) and i < len(psi_ys):
                    # Format ion name
                    ion_parts = name.split('_')
                    if len(ion_parts) >= 2:
                        element = ion_parts[0].capitalize()
                        level = roman.toRoman(int(ion_parts[1]))
                        plt.text(line_temps[i] * 1.1, psi_ys[i], f"{element} {level}", 
                                fontsize=10, verticalalignment='center')
        
        # Set plot limits and labels
        plt.ylim(10.0**low_y, 10.0**high_y)
        plt.xlim(1e4, 1e8)
        plt.xlabel('Temperature [K]')
        plt.ylabel(r'$\Psi(T) = N_e N_{\mathrm{H}} \frac{ds}{dT}$ [cm$^{-5}$ K$^{-1}$]')
        plt.title(f'{self.star_name} DEM')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/dem_{self.star_name}.png', dpi=300)
        
        return plt.gcf()