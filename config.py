from astropy import units as u

# ------------------------------
# Star Parameters
# ------------------------------
STAR_CONFIG = {
    'star_name' : 'EX',                                 # Star identifier
    'star_radius' : 0.75 * u.Rsun,                      # Stellar radius
    'star_distance' : 9.7140 * u.pc,                    # Distance to star
    'abundance' : 0.0,                                  # Abundance modifier (0.0 = solar)
}                                               

# ------------------------------
# File Paths
# ------------------------------
PATH_CONFIG = {
    'gtmat_dir' : 'gtmat',                                                      # Directory for G(T) matrices
    'output_dir' : 'output',                                                    # Directory for output files
    'plot_dir' : 'plots',                                                       # Directory for plots
    'chianti_path' : '/Users/bella/Desktop/LASP/Stars/CHIANTI_10.1_database',   # CHIANTI path
    'flux_file' : 'lines/tau-ceti.ecsv',                                        # Flux data file (must be .ecsv)
    'abundance_file' : 'sun_photospheric_2015_scott'                            # CHIANTI abundance file
}

# ------------------------------
# GTMatrix Configuration
# ------------------------------
GTMAT_CONFIG = {
    'min_wavelength': 1,                # Minimum wavelength in Angstroms
    'max_wavelength': 1500,             # Maximum wavelength in Angstroms
    'rconst': 100,                      # Spectral resolving power
    'min_templog': 4,                   # Minimum temperature in log scale
    'max_templog': 8,                   # Maximum temperature in log scale
    'npoints': 200,                     # Number of temperature points
    'pressure_list': [1e17]             # List of pressures for analysis
}

# ------------------------------
# Spectrum Configuration
# ------------------------------
SPECTRUM_CONFIG = {
    'wave_min': 5.0,                                                            # Minimum wavelength for spectrum analysis
    'wave_max': 150.0,                                                          # Maximum wavelength for spectrum analysis
    'energy_step': 5e-3,                                                        # Energy step in keV
    'flux_threshold': 1.0,                                                      # Signal-to-noise threshold
    'apply_smoothing': True,                                                    # Whether to apply smoothing
    'sigma': 1.0,                                                               # Smoothing sigma if applied
}

# ------------------------------
# DEM Configuration
# ------------------------------
DEM_CONFIG = {
    'n_walkers': 50,                    # Number of MCMC walkers
    'burn_in_steps': 200,               # Number of burn-in steps
    'production_steps': 800,            # Number of production steps
    'thread_num': 4,                    # Number of threads for parallel processing
    'init_chebyshev': [                 # Initial Chebyshev coefficients
        22.49331207,  # c₀: First Chebyshev coefficient - sets overall DEM magnitude
        -3.31678227,  # c₁: Second Chebyshev coefficient  
        -0.49848262,  # c₂: Third Chebyshev coefficient
        -1.27244452,  # c₃: Fourth Chebyshev coefficient
        -0.93897032,  # c₄: Fifth Chebyshev coefficient
        -0.67235648,  # c₅: Sixth Chebyshev coefficient
        -0.08085897   # Flux factor uncertainty (in log space)
    ]
}