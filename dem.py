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

        """
        # Check for existing results first
        if not self.load_results():
            print(f"Running MCMC for {self.star_name}")
            
            # Get flux and error data
            flux = self.gtmatrix.ion_fluxes
            err = self.gtmatrix.ion_errs

            # Get the emission line indices by calling the method
            indices = self.gtmatrix.get_emission_line_indices()
            print(indices)

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

        # return samples, lnprob, sampler
    
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
            sample_num: Number of samples to plot (default: 500)
            main_color: Color for the best-fit model (default: 'b')
            sample_color: Color for the MCMC samples (default: 'cornflowerblue')
            alpha: Transparency for sample lines (default: 0.1)
            low_y: Lower y-axis limit in log10 space (default: 19.0)
            high_y: Upper y-axis limit in log10 space (default: 26.0)
            save_path: Path to save the figure (default: None, no saving)
            
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
        
        # Get the subset of G(T) matrix for emission lines
        if hasattr(self.gtmatrix, 'emission_line_indices'):
            indices = self.gtmatrix.emission_line_indices
        else:
            indices = self.gtmatrix.get_emission_line_indices()
            
        # Print debug info
        print(f"Flux array shape: {flux_arr.shape}")
        print(f"Number of emission line indices: {len(indices)}")
        print(f"G(T) matrix shape: {self.gtmatrix.gtmat.shape}")
        
        # Extract G(T) values for the emission lines
        gofnt_matrix = np.array([self.gtmatrix.gtmat[idx] for idx in indices if idx < self.gtmatrix.gtmat.shape[0]])
        print(f"Extracted G(T) matrix shape: {gofnt_matrix.shape}")
        
        # Calculate formation temperatures for each line (temperature of maximum G(T))
        line_temps = np.array([self.temp[np.argmax(gofnt_matrix[i])] for i in range(len(gofnt_matrix))])
        print(f"Line formation temperatures: {line_temps}")
        
        # Calculate constraint values (y-axis) for each emission line
        # This is where flux / (flux_weighting * integrated G(T)) calculation happens
        # We do this directly without additional array manipulation to debug
        psi_ys = np.zeros(len(flux_arr))
        for i in range(len(flux_arr)):
            if i < len(gofnt_matrix):
                # Calculate integral of G(T) for this emission line
                integral = np.trapz(gofnt_matrix[i], self.temp)
                if integral > 0:  # Prevent division by zero
                    psi_ys[i] = flux_arr[i] / (self.flux_weighting * integral)
                    print(f"Line {i}: flux={flux_arr[i]}, integral={integral}, psi={psi_ys[i]}")
                else:
                    print(f"Warning: Line {i} has integral = {integral}")
                    psi_ys[i] = np.nan
        
        # Calculate temperature ranges for constraint bars
        temp_lows = np.zeros(len(psi_ys))
        temp_upps = np.zeros(len(psi_ys))
        
        for i in range(len(psi_ys)):
            if i < len(gofnt_matrix):
                # Calculate cumulative G(T) for determining temperature range
                gofnt_cumtrapz = np.cumsum(gofnt_matrix[i] * np.diff(self.temp)[0])
                max_val = gofnt_cumtrapz[-1]
                
                if max_val > 0:
                    # Find temperatures where cumulative G(T) reaches 16% and 84% of total
                    low_idx = np.argmin(np.abs(gofnt_cumtrapz - (0.16 * max_val)))
                    upp_idx = np.argmin(np.abs(gofnt_cumtrapz - (0.84 * max_val)))
                    
                    # Set temperature range for this line
                    temp_lows[i] = self.temp[max(0, low_idx)]
                    temp_upps[i] = self.temp[min(len(self.temp)-1, upp_idx)]
                else:
                    # Fallback to default range
                    temp_lows[i] = 1e4
                    temp_upps[i] = 1e7
                    
                print(f"Line {i}: temp range = {temp_lows[i]} - {temp_upps[i]}")
        
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
        
        # Plot flux constraints - make sure they're visible by setting high zorder and wider lines
        for i in range(len(psi_ys)):
            if np.isfinite(psi_ys[i]) and psi_ys[i] > 0:
                plt.hlines(psi_ys[i], temp_lows[i], temp_upps[i], 
                        colors='k', linewidth=2, zorder=100)
                print(f"Plotting constraint at y={psi_ys[i]}")
        
        # Add a collective label for all flux constraints
        plt.plot([], [], 'k-', linewidth=2, label='Flux Constraints')
        
        # Add ion labels if available
        if hasattr(self.gtmatrix, 'ion_list') and self.gtmatrix.ion_list is not None:
            ion_names = self.gtmatrix.ion_list
            
            for i, name in enumerate(ion_names):
                if i < len(line_temps) and i < len(psi_ys) and np.isfinite(psi_ys[i]) and psi_ys[i] > 0:
                    # Parse ion name
                    ion_parts = name.split('_')
                    if len(ion_parts) >= 2:
                        try:
                            # Format name: capitalize element and convert to Roman numeral
                            element = ion_parts[0].capitalize()
                            level = roman.toRoman(int(ion_parts[1]))
                            formatted_name = f"{element} {level}"
                            
                            # Add text label next to constraint line
                            plt.text(line_temps[i] * 1.1, psi_ys[i], formatted_name,
                                    fontsize=10, verticalalignment='center')
                            
                            print(f"Adding label '{formatted_name}' at {line_temps[i]}, {psi_ys[i]}")
                        except (ValueError, TypeError) as e:
                            print(f"Error formatting ion name {name}: {e}")
        
        # Set plot limits and labels
        plt.ylim(10.0**low_y, 10.0**high_y)
        plt.xlim(1e4, 1e8)  # Temperature range from 10^4 to 10^8 K
        plt.xlabel('Temperature [K]')
        y_label = r'$\Psi(T) = N_e N_{\mathrm{H}} \frac{ds}{dT}$ '
        y_label += r'[cm$^{-5}$ K$^{-1}$]'
        plt.ylabel(y_label)
        plt.title(f'{self.star_name} DEM')
        plt.legend()
        plt.tight_layout()
        plt.show()

        save_path = f'plots/dem_{self.gtmatrix.star_name}.png'
        
        plt.savefig(save_path, dpi=300)
        print(f"DEM plot saved to {save_path}")
        
        return plt.gcf()