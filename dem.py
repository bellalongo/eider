from fitting import MCMCFitter, MCMCConfig
from probability_fxns import ln_likelihood_dem
import numpy as np
from astropy import units as u
from typing import Tuple, Any
import multiprocessing
from os.path import exists

class DEM:
    def __init__(self, 
                 gtmatrix, 
                 radius: u.Quantity, 
                 distance: u.Quantity):
        """

        """
        self.gtmatrix = gtmatrix
        self.radius = radius 
        self.distance = distance

        print(len(gtmatrix.ion_fluxes), len(gtmatrix.ion_errs))

        # MCMC result files
        self.sample_dir = f'mcmc/samples_{self.gtmatrix.star_name}'
        self.lnprob_dir = f'mcmc/lnprob_{self.gtmatrix.star_name}'
        
        # Temperature grid
        self.temp = np.logspace(4, 8, 200)
        self.log_temp = np.log10(self.temp)

        # Initial Chebyshev coefficient guesses
        self.init_chebyshev = [
            22.49331207,  # c₀: First Chebyshev coefficient - sets overall DEM magnitude
            -3.31678227,  # c₁: Second Chebyshev coefficient  
            -0.49848262,  # c₂: Third Chebyshev coefficient
            -1.27244452,  # c₃: Fourth Chebyshev coefficient
            -0.93897032,  # c₄: Fifth Chebyshev coefficient
            -0.67235648,  # c₅: Sixth Chebyshev coefficient
            -0.08085897   # Flux factor uncertainty (in log space)
        ]
        
        # Calculate flux weighting based on stellar geometry
        self.flux_weighting = ((np.pi * u.sr * (self.radius**2.0) * 
                               (1.0 / (self.distance**2.0))).to(u.sr)).value
        
        # Check if MCMC has already been performed
        if exists(self.sample_dir) and exists(self.lnprob_dir):
            self.samples = np.load(self.sample_dir)
            self.lnprob = np.load(self.lnprob_dir)
        else:
            self.samples, self.lnprob, _ = self.run_mcmc(self.gtmatrix.ion_fluxes, self.gtmatrix.ion_errs)


    def run_mcmc(self, 
                 flux: np.ndarray,
                 err: np.ndarray,
                 nwalkers: int = 50,
                 burn_in_steps: int = 200,
                 production_steps: int = 800,
                 thread_num: int = multiprocessing.cpu_count(),
                 progress_interval: int = 100) -> Tuple[np.ndarray, np.ndarray, Any]:
        """

        """
        # Create MCMC configuration
        config = MCMCConfig(
            n_walkers=nwalkers,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            thread_num=thread_num,
            progress_interval=progress_interval
        )
        
        # Initialize fitter
        fitter = MCMCFitter(config)
        
        # Package likelihood arguments
        likelihood_args = [
            flux,
            err, 
            self.log_temp,
            self.temp,
            self.gtmatrix.gtmat,
            self.flux_weighting
        ]

        # Run MCMC
        samples, lnprob, sampler = fitter.fit(
            init_pos=self.init_chebyshev,
            likelihood_func=ln_likelihood_dem,
            likelihood_args=likelihood_args
        )

        np.save(self.sample_dir, samples)
        np.save(self.lnprob_dir, lnprob)

        return samples, lnprob, sampler