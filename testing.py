import os

# Set environment variable before any other imports
if 'XUVTOP' not in os.environ:
    os.environ['XUVTOP'] = '/Users/bella/Desktop/LASP/Stars/CHIANTI_10.1_database'
    print(f"XUVTOP environment set to {os.environ['XUVTOP']}")

from utils import *
from gtmatrix import *
from spectrum import *
from dem import *
from config import *

def main():
    # Create output directories
    create_directories(PATH_CONFIG)

    # Create GTMatrix 
    gtmatrix = GTMatrix(STAR_CONFIG, GTMAT_CONFIG, PATH_CONFIG)
    gtmatrix.initialize()
    gtmatrix.load_line_data()
    gtmatrix.generate_all_matrices()
    gtmatrix.generate_heatmap()

    # Create DEM
    dem = DEM(gtmatrix, DEM_CONFIG, STAR_CONFIG)
    
    # Run MCMC (or load existing results)
    samples, lnprob, _ = dem.run_mcmc()
    dem.plot_dem()
    dem.create_corner_plot()

    # Create Spectrum object
    spectrum = Spectrum(dem)
    
    # # Generate and plot spectrum
    # spectrum.generate_spectrum(sample_num=1000) # -> Add me to config !
    # spectrum.plot_spectrum(save_path=PATH_CONFIG) # -> add me to path config !
    # plt.show()

    print("All plots have been generated!")
    
    # Output suggestions for further parameter tuning
    best_idx = np.argmax(lnprob)
    best_params = samples[best_idx]
    print("\nBest-fit parameters:")
    for i, param in enumerate(best_params[:-1]):
        print(f"c{i}: {param:.4f}")
    print(f"Flux factor: {best_params[-1]:.4f}")
    
    print("\nSuggested improvements for next run:")
    # Check if the best-fit parameters are close to the initial values
    init_params = np.array(DEM_CONFIG['init_chebyshev'])
    diff = np.abs(best_params - init_params)
    if np.any(diff > 1.0):
        print("- Try using the best-fit parameters as initial values for the next run:")
        print(f"  'init_chebyshev': {best_params.tolist()}")
    
    # Check if the best-fit is at the edge of the prior range
    if abs(best_params[0]) > 24.0:
        print("- The baseline magnitude (c0) is near the edge of the prior range.")
        print("  Consider adjusting the prior range in probability_fxns.py.")
    
    print("\nDone!")

    

if __name__ == '__main__':
    main()

    print('All done!')