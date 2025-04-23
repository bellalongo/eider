from typing import List, Union
from numpy.polynomial.chebyshev import chebval
import numpy as np
from tinygp import kernels, GaussianProcess
from jax import numpy as jnp

from eider.gtmatrix import *

class DEMPriors:
    """
        This class contains static methods implementing prior probability functions
        for Differential Emission Measure (DEM) models. These priors enforce physical
        constraints and regularization on DEM model parameters.
    """
    @staticmethod
    def chebyshev_prior(params: List[float],
                        psi_low: float,
                        psi_high: float) -> float:
        """
            Calculates the prior probability for a Chebyshev polynomial DEM model.
            Enforces physical constraints on coefficient magnitudes and applies regularization
            to prevent unrealistic oscillations in the DEM.
            Arguments:
                params (list): List of model parameters. The last element is the flux factor,
                             and the rest are Chebyshev polynomial coefficients.
                psi_low (float): Lower bound for DEM magnitude in log10 space
                psi_high (float): Upper bound for DEM magnitude in log10 space
            Returns:
                float: Log prior probability. 0.0 for acceptable models, -np.inf for rejected models,
                      or negative values for penalized models
        """
        flux_factor = params[-1]
        coeffs = params[:-1]
        lp = 0.0
        
        # Add penalties based on coefficient values
        if coeffs[0] >= psi_high:
            lp += psi_high - coeffs[0]
        elif coeffs[0] <= psi_low:
            lp += coeffs[0] - psi_low
        
        # Check polynomial value at 0
        if chebval(0.0, coeffs) <= 15.0:  # Lower threshold for Tau Ceti
            return -np.inf
        
        # Check coefficient magnitudes
        for coeff in coeffs:
            if not (-100.0 <= coeff <= 100.0):
                return -np.inf
        
        # Check flux factor bounds
        if not (-2.0 <= flux_factor <= 2.0):  # Wider range for flexibility
            return -np.inf
        
        # Check polynomial behavior at endpoints
        if (chebval(-1.0, coeffs) <= chebval(-0.99, coeffs) or
            chebval(1.0, coeffs) >= chebval(0.99, coeffs)):
            return -np.inf
        
        # Add regularization to prevent extreme oscillations -> MAYBE DELETE
        oscillation_penalty = 0.0
        for i in range(1, len(coeffs)-1):
            if abs(coeffs[i]) > 20 and abs(coeffs[i+1]) > 20:
                if coeffs[i] * coeffs[i+1] < 0:  # Opposite signs
                    oscillation_penalty -= abs(coeffs[i] * coeffs[i+1]) / 1000.0
        
        return lp + oscillation_penalty
    
    
    @staticmethod
    def gp_prior(params: List[float], 
                 psi_low: float,
                 psi_high: float) -> float:
        """
            Calculates the prior probability for a Gaussian Process DEM model.
            Enforces constraints on hyperparameters and knot values.
            Arguments:
                params (list): List of model parameters. The last three elements are GP 
                             hyperparameters, and the rest are knot values.
                psi_low (float): Lower bound for DEM magnitude in log10 space
                psi_high (float): Upper bound for DEM magnitude in log10 space
            Returns:
                float: Log prior probability. 0.0 for acceptable models, -np.inf for rejected models
        """
        # Check if all parameters are infinite
        if not all(np.isfinite(p) for p in params):
            return -np.inf
        
        # Grab the gaussian prior model parameters from chebyshev params
        amp1 = np.exp(params[-3])
        amp2 = np.exp(params[-2])
        scale = params[-1]
        knots = params[:-3]

        # Check hyperparameter bounds
        if not (0.1 <= scale <= 10.0):
            return -np.inf
        if not (0.01 <= amp1 <= 100.0):
            return -np.inf
        if not (0.01 <= amp2 <= 100.0):
            return -np.inf
        
        # Check knot ordering constraints
        if knots[0] < knots[1] or knots[-1] > knots[-2]:
            return -np.inf
        
        # Check knot value bounds
        if not (psi_low <= knots[0] <= psi_high):
            return -np.inf
        if any(not (psi_low <= k <= psi_high) for k in knots[:-1]):
            return -np.inf
        
        return 0.0
    

class DEMLikelihood:
    """
        This class contains static methods implementing likelihood functions
        for Differential Emission Measure (DEM) models. These functions calculate
        how well a given DEM model explains the observed emission line fluxes.
    """
    @staticmethod
    def chebyshev_likelihood(params: List[float], 
                             y: np.ndarray,
                             yerr: Union[np.ndarray, float],
                             log_temp: np.ndarray, 
                             temp: np.ndarray,
                             gtmat: np.ndarray,
                             flux_weighting: float) -> float:
        """
            Calculates the log likelihood for a Chebyshev polynomial DEM model
            given observed fluxes and a G(T) matrix.
            Arguments:
                params (list): List of model parameters. The last element is the flux factor,
                             and the rest are Chebyshev polynomial coefficients.
                y (ndarray): Observed emission line fluxes
                yerr (ndarray or float): Uncertainties on observed fluxes
                log_temp (ndarray): Log10 of temperature grid
                temp (ndarray): Temperature grid
                gtmat (ndarray): G(T) matrix with shape (n_lines, n_temperatures)
                flux_weighting (float): Scaling factor for flux calculations
            Returns:
                float: Log likelihood value, or -np.inf for invalid models
        """
        # Calculate model DEM
        flux_factor = 10.0**(params[-1]) 
        coeffs = params[:-1]
        shift_log_temp = (log_temp - np.mean(log_temp)) / (0.5 * (np.max(log_temp) - np.min(log_temp)))
        psi_model = 10.0**chebval(shift_log_temp, coeffs)
        
        if np.nanmin(psi_model) <= 0:
            return -np.inf
            
        # Calculate model spectrum
        model = GTMatrix.calculate_integral(gtmat, temp, psi_model, flux_weighting)
        
        # Calculate log likelihood
        var_term = (flux_factor * model)**2 + yerr**2
        log_likelihood = np.sum(
            np.log(1.0 / np.sqrt(2.0 * np.pi * var_term)) - 
            0.5 * (y - model)**2 / var_term
        )
        
        return log_likelihood if np.isfinite(log_likelihood) else -np.inf


    @staticmethod
    def gp_likelihood(params: List[float], 
                      knot_locs: np.ndarray,
                      y: np.ndarray, 
                      yerr: Union[np.ndarray, float],
                      temp: np.ndarray, 
                      gtmat: np.ndarray,
                      flux_weighting: float) -> float:
        """
            Calculates the log likelihood for a Gaussian Process DEM model
            given observed fluxes and a G(T) matrix.
            Arguments:
                params (list): List of model parameters. The last three elements are GP 
                             hyperparameters, and the rest are knot values.
                knot_locs (ndarray): Locations of knots in normalized temperature space
                y (ndarray): Observed emission line fluxes
                yerr (ndarray or float): Uncertainties on observed fluxes
                temp (ndarray): Temperature grid
                gtmat (ndarray): G(T) matrix with shape (n_lines, n_temperatures)
                flux_weighting (float): Scaling factor for flux calculations
            Returns:
                float: Log likelihood value, or -np.inf for invalid models 
        """
        try:
            # Set up GP
            amp1, amp2 = np.exp(params[-3]), np.exp(params[-2])
            scale = params[-1]
            knots = jnp.array(params[:-3])
            kernel = amp2 * kernels.Matern32(scale) + kernels.Constant(amp1)
            gp = GaussianProcess(kernel, jnp.array(knot_locs))
            
            # Predict DEM
            x_test = jnp.linspace(-1, 1, len(temp))
            mu, cov = gp.predict(knots, x_test, return_var=True)
            cov = np.array(cov)
            cov[cov <= 0.0] = np.min(cov[cov > 0.0])
            
            # Sample DEM and calculate model
            psi_model = 10.0**np.random.normal(loc=mu, scale=np.sqrt(cov))
            model = GTMatrix.calculate_integral(psi_model, gtmat, temp, flux_weighting)
            
            # Calculate log likelihood
            var_term = yerr**2
            log_likelihood = np.sum(
                np.log(1.0 / np.sqrt(2.0 * np.pi * var_term)) -
                0.5 * (y - model)**2 / var_term
            )
            
            return log_likelihood if np.isfinite(log_likelihood) else -np.inf
            
        except ValueError:
            return -np.inf


@staticmethod
def ln_likelihood_dem(params: List[float], 
                      psi_low : float,
                      psi_high : float,
                      y: np.ndarray,
                      yerr: Union[np.ndarray, float],
                      log_temp: np.ndarray, temp: np.ndarray,
                      gtmat: np.ndarray,
                      flux_weighting: float) -> float:
    """
        Calculates the full posterior probability (prior × likelihood) for a 
        Chebyshev polynomial DEM model. This is the main function used by MCMC sampling.
        Arguments:
            params (list): List of model parameters. The last element is the flux factor,
                         and the rest are Chebyshev polynomial coefficients.
            psi_low (float): Lower bound for DEM magnitude in log10 space
            psi_high (float): Upper bound for DEM magnitude in log10 space
            y (ndarray): Observed emission line fluxes
            yerr (ndarray or float): Uncertainties on observed fluxes
            log_temp (ndarray): Log10 of temperature grid
            temp (ndarray): Temperature grid
            gtmat (ndarray): G(T) matrix with shape (n_lines, n_temperatures)
            flux_weighting (float): Scaling factor for flux calculations
        Returns:
            float: Log posterior probability (prior + likelihood)
    """
    # Define log prior
    lp = DEMPriors.chebyshev_prior(params, psi_low, psi_high)

    # Return infinite log priors
    if np.isfinite(lp):
        return lp + DEMLikelihood.chebyshev_likelihood(
            params, y, yerr, log_temp, temp, gtmat, flux_weighting
        )
    return -np.inf


@staticmethod
def ln_likelihood_gp(params: List[float], 
                     psi_low : float,
                     psi_high : float,
                     knot_locs: np.ndarray,
                     y: np.ndarray, 
                     yerr: Union[np.ndarray, float],
                     temp: np.ndarray, 
                     gtmat: np.ndarray,
                     flux_weighting: float) -> float:
    """
        Calculates the full posterior probability (prior × likelihood) for a 
        Gaussian Process DEM model. This is the main function used by MCMC sampling.
        Arguments:
            params (list): List of model parameters. The last three elements are GP 
                         hyperparameters, and the rest are knot values.
            psi_low (float): Lower bound for DEM magnitude in log10 space
            psi_high (float): Upper bound for DEM magnitude in log10 space
            knot_locs (ndarray): Locations of knots in normalized temperature space
            y (ndarray): Observed emission line fluxes
            yerr (ndarray or float): Uncertainties on observed fluxes
            temp (ndarray): Temperature grid
            gtmat (ndarray): G(T) matrix with shape (n_lines, n_temperatures)
            flux_weighting (float): Scaling factor for flux calculations
        Returns:
            float: Log posterior probability (prior + likelihood)
    """
    # Define log prior
    lp = DEMPriors.gp_prior(params, psi_low, psi_high)

    # Return infinite log priors
    if np.isfinite(lp):
        return lp + DEMLikelihood.gp_likelihood(
            params, knot_locs, y, yerr, temp, gtmat, flux_weighting
        )
    return -np.inf