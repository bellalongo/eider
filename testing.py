from utils import *
from gtmatrix import *
from spectrum import *
from dem import *
from config import *

from astropy import units as u


def main():
    # Check environment
    check_environment()

    # Create output directories
    create_directories(PATH_CONFIG)

    gtmatrix = GTMatrix(STAR_CONFIG, GTMAT_CONFIG)

    # Create GTMatrix 
    gtmatrix.initialize()
    gtmatrix.load_line_data()
    gtmatrix.generate_all_matrices()
    gtmatrix.generate_heatmap()

    


if __name__ == '__main__':
    main()

    print('All done!')