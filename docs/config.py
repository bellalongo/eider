from astropy import units as u

# ------------------------------
# Star Parameters
# star_name : Star identifier
# star_radius : Stellar radius (parsecs)
# star_distance : Distance to star
# abundance : Abundance modifier (0.0 = solar) -> log
# ------------------------------
STAR_CONFIG = {
    'star_name' : 'Tau-Ceti',
    'star_radius' : 0.793 * u.Rsun,
    'star_distance' : 3.65 * u.pc,                   
    'abundance' : -0.3
}                                               

# ------------------------------
# File Paths
# gtmat_dir : Directory for G(T) matrices
# output_dir : Directory for output files
# plot_dir : Directory for plots
# chianti_path : CHIANTI path
# flux_file : Flux data file (must be .ecsv)
# abundance_file : CHIANTI abundance file 
# ------------------------------
PATH_CONFIG = {
    'gtmat_dir' : 'gtmat',                                                      
    'output_dir' : 'output',                                                    
    'plot_dir' : 'plots',                                                       
    'chianti_path' : '/Users/bella/Desktop/LASP/Stars/CHIANTI_10.1_database',  
    'flux_file' : 'lines/wasp-121.ecsv',                                        
    'abundance_file' : 'sun_photospheric_2015_scott'                       
}

# ------------------------------
# GTMatrix Configuration
# min_wavelength : Minimum wavelength in Angstroms
# max_wavelength : Maximum wavelength in Angstroms
# rconst : Spectral resolving power
# min_templog : Minimum temperature in log scale
# max_templog : Maximum temperature in log scale
# npoints : Number of temperature points
# pressure_list : List of pressures for analysis
# ------------------------------
GTMAT_CONFIG = {
    'min_wavelength': 1,                
    'max_wavelength': 1700,            
    'rconst': 100,                      
    'min_templog': 3,                  
    'max_templog': 8,                   
    'npoints': 200,                    
    'pressure_list': [1e17]             
}

# ------------------------------
# Spectrum Configuration
# wave_min : Minimum wavelength for spectrum analysis
# wave_max : Maximum wavelength for spectrum analysis
# energy_step : Energy step in keV
# flux_threshold : Signal-to-noise threshold
# apply_smoothing : Whether to apply smoothing
# sigma : Smoothing sigma if applied
# ------------------------------
SPECTRUM_CONFIG = {
    'wave_min': 5.0,                                                           
    'wave_max': 150.0,                                                  
    'energy_step': 5e-3,                                                     
    'flux_threshold': 1.0,                                                
    'apply_smoothing': True,                                                
    'sigma': 1.0,                                                            
}

# ------------------------------
# DEM Configuration
# n_walkers : Number of MCMC walkers
# burn_in_steps : Number of burn-in steps
# production_steps : Number of production steps
# thread_num : Number of threads for parallel processing
# progress_interval :
# init_chebyshev # Initial Chebyshev coefficients
#   c₀: First Chebyshev coefficient - sets overall DEM magnitude
#   c₁: Second Chebyshev coefficient  
#   c₂: Third Chebyshev coefficient
#   c₃: Fourth Chebyshev coefficient
#   c₄: Fifth Chebyshev coefficient
#   c₅: Sixth Chebyshev coefficient
#   Flux factor uncertainty (in log space)
# psi_low : Low pressure value for DEM fit (in log space)
# psi_high : High pressure value for DEM fit (in log space)
# ------------------------------
DEM_CONFIG = {
    'n_walkers': 100,                   
    'burn_in_steps': 200,            
    'production_steps': 800,       
    'thread_num': 4,              
    'progress_interval': 100,        
    'init_chebyshev': [               
        22.49331207,                
        -3.31678227,            
        -0.49848262,                   
        -1.27244452,                    
        -0.93897032,                   
        -0.67235648,                   
        -0.08085897                    
    ],
    'psi_low': 19,
    'psi_high': 25
}
