from gtmatrix import *
# from dem import *

from astropy import units as u


def main():
    # !NOTE BEFORE RUNNING: export XUVTOP='/Users/bella/Desktop/LASP/Stars/CHIANTI_10.1_database'
    star_name = 'EX'
    abundance_file = 'sun_photospheric_2015_scott'
    flux_file = 'tau-ceti.ecsv' # Must be ecsv file

    # Stellar parameters
    abundance = 0.0
    radius = 0.75 * u.Rsun
    distance = 9.7140 * u.pc
    
    # Directory defining
    gtmat_dir = 'gtmat' # G(T) matrix directory  
    
    # Generate G(t) matrix
    gtmat = GTMatrix(star_name, abundance, abundance_file, flux_file, gtmat_dir)

    # Generate DEM
    dem = DEM(gtmat, radius, distance)


    print('Done')
    
    
if __name__ == '__main__':
    main()

    print('All done!')