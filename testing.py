from utils import *
from gtmatrix import *
from spectrum import *
from dem import *
from config import *

from astropy import units as u


def main():
    # Check environment
    check_environment(PATH_CONFIG)

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
    samples, lnprob, sampler = dem.run_mcmc()

    # Create Spectrum object
    spectrum = Spectrum(dem)
    
    # Generate and plot spectrum
    spectrum.generate_spectrum(sample_num=1000) # -> Add me to config !
    spectrum.plot_spectrum(save_path=PATH_CONFIG) # -> add me to path config !
    plt.show()

    

if __name__ == '__main__':
    main()

    print('All done!')