import numpy as np
import emcee
import multiprocessing
from typing import List, Callable, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


@dataclass
class MCMCConfig:
   """

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

    """
    def __init__(self, config):
       """
                
       """
       self.config = config
       self._sampler = None

    def initialize_walkers(self, init_pos: np.ndarray) -> List[np.ndarray]:
        """
            
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
            
        """
        print("Running burn-in...")
        
        # Use tqdm to create a progress bar for the burn-in phase
        with tqdm(total=steps, desc="Burn-in", unit="step") as pbar:
            # Define a callback for progress updates
            def update_progress(state, *args):
                pbar.update(1)
            
            # Run the MCMC with progress tracking
            p0, prob, _ = sampler.run_mcmc(pos, steps, progress=update_progress)
        
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
            
        """
        print("Running production...")
        
        # Use tqdm to create a progress bar for the production phase
        with tqdm(total=steps, desc="Production", unit="step") as pbar:
            # Define a callback for progress updates
            def update_progress(state, *args):
                pbar.update(1)
            
            # Run the MCMC with progress tracking
            sampler.run_mcmc(pos, steps, progress=update_progress)

    def fit(self, 
            init_pos: np.ndarray,
            likelihood_func: Callable,
            likelihood_args: List[Any]
            ) -> Tuple[np.ndarray, np.ndarray, emcee.EnsembleSampler]:
        """
            
        """
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