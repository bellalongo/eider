import os
import sys

sys.path.insert(0, '../') 

# Set environment variable before any other imports
if 'XUVTOP' not in os.environ:
    os.environ['XUVTOP'] = '/Users/bella/Desktop/LASP/Stars/CHIANTI_10.1_database'
    print(f"XUVTOP environment set to {os.environ['XUVTOP']}")

import eider
from config import *


def main():
    # Create output directories
    eider.create_directories(PATH_CONFIG)
    eider.utils.check_environment(PATH_CONFIG)

    # Create GTMatrix 
    gtmatrix = eider.GTMatrix(STAR_CONFIG, GTMAT_CONFIG, PATH_CONFIG)
    gtmatrix.initialize()
    gtmatrix.load_line_data()
    gtmatrix.generate_all_matrices()
    gtmatrix.generate_heatmap()

    # Create DEM
    dem = eider.DEM(gtmatrix, DEM_CONFIG, STAR_CONFIG)
    
    # Run MCMC (or load existing results)
    samples, lnprob, _ = dem.run_mcmc()
    dem.plot_dem()
    dem.create_corner_plot()

    # # Create Spectrum object
    # spectrum = Spectrum(dem)
    
    # # Generate and plot spectrum
    # spectrum.generate_spectrum(sample_num=1000) # -> Add me to config !
    # spectrum.plot_spectrum(save_path=PATH_CONFIG) # -> add me to path config !
    # plt.show()

if __name__ == '__main__':
    main()

    print('All done!')