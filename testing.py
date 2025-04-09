import os

from utils import *
from gtmatrix import *
from spectrum import *
from dem import *
from config import *


def main():
    # Set environment variable before any other imports
    if 'XUVTOP' not in os.environ:
        os.environ['XUVTOP'] = '/Users/bella/Desktop/LASP/Stars/CHIANTI_10.1_database'
        print(f"XUVTOP environment set to {os.environ['XUVTOP']}")

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
    dem.run_mcmc()
    dem.plot_dem()

    # Create Spectrum object
    spectrum = Spectrum(dem)
    
    # # Generate and plot spectrum
    # spectrum.generate_spectrum(sample_num=1000) # -> Add me to config !
    # spectrum.plot_spectrum(save_path=PATH_CONFIG) # -> add me to path config !
    # plt.show()

    

if __name__ == '__main__':
    main()

    print('All done!')