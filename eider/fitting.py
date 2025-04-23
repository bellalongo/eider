import numpy as np
import emcee
import multiprocessing
from typing import List, Callable, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class MCMCConfig:
   """
        Configuration class for MCMC sampling parameters.
        This class stores all settings required for configuring an MCMC sampler,
        including number of walkers, steps, and parallel processing options.
        Attributes:
            n_walkers (int): Number of walkers for the ensemble sampler
            burn_in_steps (int): Number of steps for the burn-in phase
            production_steps (int): Number of steps for the production phase
            init_spread (float): Initial spread factor for walker positions
            second_spread (float): Spread factor after burn-in phase
            double_burn (bool): Whether to perform a second burn-in phase
            thread_num (int): Number of threads for parallel processing
            progress_interval (int): Interval for progress updates
   """
   n_walkers: int
   burn_in_steps: int 
   production_steps: int
   init_spread: float = 1e-1
   second_spread: float = 1e-2
   double_burn: bool = True
   thread_num: int = multiprocessing.cpu_count()
   progress_interval: int = 100


class MCMCFitter:
    """
        Class for performing Markov Chain Monte Carlo (MCMC) parameter fitting.
        This class wraps the emcee package to simplify the MCMC workflow, including
        initialization, burn-in, and production phases with parallel processing support.
        Attributes:
            config (MCMCConfig): Configuration object containing MCMC parameters
            _sampler (emcee.EnsembleSampler): Internal reference to the emcee sampler
    """
    def __init__(self, config):
       """
            Initializes the MCMCFitter with a configuration object.
            Arguments:
                config (MCMCConfig): Configuration object containing MCMC parameters
            Returns:
                None
       """
       self.config = config
       self._sampler = None

    def initialize_walkers(self, init_pos: np.ndarray) -> List[np.ndarray]:
        """
            Initializes the walker positions around the initial position.
            Creates an ensemble of walker positions by adding random noise to the
            initial position, scaled by the init_spread factor.
            Arguments:
                init_pos (ndarray): Initial position vector for the parameters
            Returns:
                list: List of walker position arrays
        """
        # Get the number of dimensions in the initial position
        ndim = len(init_pos)

        return [
            init_pos + self.config.init_spread * np.random.randn(ndim) * init_pos
            for _ in range(self.config.n_walkers)
        ]
    
    def run_burn_in(self, 
                    pos: List[np.ndarray], 
                    sampler: emcee.EnsembleSampler,
                    steps: int
                    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
            Runs the burn-in phase of MCMC sampling.
            This phase allows the walkers to reach the high-probability region
            of the parameter space before collecting samples for analysis.
            Arguments:
                pos (list): Initial positions of walkers
                sampler (emcee.EnsembleSampler): The emcee sampler object
                steps (int): Number of burn-in steps to run
            Returns:
                tuple: (new_positions, log_probabilities) after burn-in
        """
        print("Running burn-in...")
        
        # Use emcee's built-in progress monitoring
        p0, prob, _ = sampler.run_mcmc(pos, steps, progress=True)
        
        # Reset positions around highest probability position
        best_pos = p0[np.argmax(prob)]
        n_walkers = len(pos)
        ndim = len(best_pos)
        
        p0 = [
            best_pos + self.config.second_spread * np.random.randn(ndim) * best_pos
            for _ in range(n_walkers)
        ]
        
        sampler.reset()
        return p0, prob
    
    def run_production(self, 
                        pos: List[np.ndarray], 
                        sampler: emcee.EnsembleSampler,
                        steps: int
                        ) -> None:
        """
            Runs the production phase of MCMC sampling.
            This is the main sampling phase where the walker positions are
            recorded for posterior analysis.
            Arguments:
                pos (list): Initial positions of walkers
                sampler (emcee.EnsembleSampler): The emcee sampler object
                steps (int): Number of production steps to run
            Returns:
                None
        """
        print("Running production...")
        
        # Use emcee's built-in progress monitoring
        sampler.run_mcmc(pos, steps, progress=True)

    def fit(self, 
            init_pos: np.ndarray,
            likelihood_func: Callable,
            likelihood_args: List[Any]
            ) -> Tuple[np.ndarray, np.ndarray, emcee.EnsembleSampler]:
        """
            Performs the complete MCMC fitting process.
            This is the main method that combines walker initialization,
            burn-in phase(s), and production run with parallel processing.
            Arguments:
                init_pos (ndarray): Initial position vector for the parameters
                likelihood_func (callable): Function to calculate log probability
                likelihood_args (list): Arguments to pass to the likelihood function
            Returns:
                tuple: (flatchain, flatlnprobability, sampler) containing the flattened
                      chain of samples, log probabilities, and the sampler object        """
        print(f'Starting ln_likelihood: {likelihood_func(init_pos, *likelihood_args)}')
        
        # Initialize walkers
        pos = self.initialize_walkers(init_pos)
        
        # Set up parallel processing
        with ThreadPoolExecutor(max_workers=self.config.thread_num) as executor:
            sampler = emcee.EnsembleSampler(
                self.config.n_walkers, len(init_pos),
                likelihood_func, args=likelihood_args,
                pool=executor
            )
            
            # First burn-in
            pos, _ = self.run_burn_in(pos, sampler, self.config.burn_in_steps)
            
            # Optional second burn-in
            if self.config.double_burn:
                pos, _ = self.run_burn_in(pos, sampler, self.config.burn_in_steps)
                
            # Production run
            self.run_production(pos, sampler, self.config.production_steps)
            
        return sampler.flatchain, sampler.flatlnprobability, sampler