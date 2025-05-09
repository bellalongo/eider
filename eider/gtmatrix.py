from astropy.table import Table
import roman
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ChiantiPy.core as ch
from ChiantiPy.tools.io import masterListRead
from tqdm import tqdm
from os.path import exists
from typing import Tuple, List, Optional, Union, Dict


# ------------------------------
# GTMatrix Class
# ------------------------------
class GTMatrix:
    """
        This class handles the generation and management of G(T) contribution matrices
        for stellar emission line analysis. It processes emission lines from CHIANTI atomic
        database and calculates the temperature-dependent contribution functions.
        Attributes:
            star_config (dict): Configuration for the target star
            gtmat_config (dict): Configuration for G(T) matrix generation
            path_config (dict): Configuration for file paths
            star_name (str): Name of the target star
            abundance (float): Abundance modifier relative to solar
            abundance_file (str): Name of the abundance file to use with CHIANTI
            min_wavelength (float): Minimum wavelength in Angstroms for analysis
            max_wavelength (float): Maximum wavelength in Angstroms for analysis
            rconst (float): Spectral resolving power for wavelength grid
            min_templog (float): Minimum temperature in log scale
            max_templog (float): Maximum temperature in log scale
            npoints (int): Number of temperature points
            pressure_list (list): List of pressures for analysis
            temp (ndarray): Array of temperature values
            wave_arr (ndarray): Array of wavelength bin centers
            bin_arr (ndarray): Array of wavelength bin widths
            gtmat (ndarray): The G(T) matrix with shape (n_wavelengths, n_temperatures)
            ion_list (ndarray): Array of unique ion names in CHIANTI format
            ion_fluxes (ndarray): Array of observed fluxes for each ion
            ion_errs (ndarray): Array of flux errors for each ion

    """
    def __init__(self, 
                 star_config, 
                 gtmat_config, 
                 path_config):
        """
            Initializes the GTMatrix object with configuration settings for star,
            G(T) matrix generation, and file paths.
            Arguments:
                star_config (dict): Configuration for the target star
                gtmat_config (dict): Configuration for G(T) matrix generation
                path_config (dict): Configuration for file paths
            Returns:
                None
        """
        self.star_config = star_config
        self.gtmat_config = gtmat_config
        self.path_config = path_config

        # Load the config data
        self._load_data()

        # Determine abundance type
        self.abundance_type = self._get_abund_type()
        
        # Initialize data containers, will be populated by method calls
        self.temp = None
        self.wave_arr = None
        self.bin_arr = None
        self.gtmat_str = None
        self.ions = None
        self.reg_ions = None
        self.chianti_table = None
        self.ion_list = None
        self.ion_fluxes = None
        self.ion_errs = None
        self.gtmat = None

    # ------------------------------
    # Private Helper Methods
    # ------------------------------
    def _load_data(self):
        """
            Extracts parameters from the configuration dictionaries and assigns them
            to instance variables. This includes star parameters, G(T) matrix parameters,
            and file paths.
            Arguments:
                None
            Returns:
                None
        """
        # Extract star parameters from star config
        self.star_name = self.star_config['star_name'].lower()
        self.abundance = self.star_config['abundance']
        
        # Extract GTMatrix parameters from config
        self.min_wavelength = self.gtmat_config['min_wavelength']
        self.max_wavelength = self.gtmat_config['max_wavelength']
        self.rconst = self.gtmat_config['rconst']
        self.min_templog = self.gtmat_config['min_templog']
        self.max_templog = self.gtmat_config['max_templog']
        self.npoints = self.gtmat_config['npoints']
        self.pressure_list = self.gtmat_config['pressure_list']

        # Extract parameters from path config
        self.flux_file = self.path_config['flux_file']
        self.gtmat_dir = self.path_config['gtmat_dir']
        self.abundance_file = self.path_config['abundance_file']

    def _get_abund_type(self) -> str:
        """
            Determines the abundance type string based on the abundance value.
            This is used for file naming conventions.
            Arguments:
                None
            Returns:
                str: Abundance type string ('sol0', 'sub1', or 'sup1')
        """
        if self.abundance == 0.0:
            return 'sol0'
        elif self.abundance == -1.0:
            return 'sub1'
        elif self.abundance == 1.0:
            return 'sup1'
        else:
            # Default to solar if not a standard value
            return 'sol0'

    def _gen_rconst_arr(self) -> Tuple[np.ndarray, np.ndarray]:
        """
            Generates wavelength arrays with constant resolving power.
            Creates an array of wavelength bin centers and bin widths.
            Arguments:
                None
            Returns:
                tuple: (wave_arr, bin_arr) arrays of wavelength bin centers and widths
        """
        # Create wavelength and bin arrays
        wave_arr = []
        bin_arr = []

        # Calculate wavelength points with constant resolving power
        dlambda_min = self.min_wavelength / self.rconst
        for temp_wave in np.arange(self.min_wavelength, 
                                   self.max_wavelength + dlambda_min, 
                                   dlambda_min):
            temp_dlambda = temp_wave / self.rconst
            bin_arr.append(temp_dlambda)
            wave_arr.append(temp_wave + (0.5 * temp_dlambda))

        return np.array(wave_arr), np.array(bin_arr)
    
    def _create_chianti_table(self) -> Table:
        """
            Creates an Astropy Table from the input ECSV file and converts ion names
            to CHIANTI format (e.g., 'C III' to 'c_3').
            Arguments:
                None
            Returns:
                Table: Astropy Table with reformatted ion names
            Raises:
                ValueError: If the ECSV file cannot be read or processed
        """  
        try:
            # Read from the ecsv file
            table = Table.read(self.flux_file, format='ascii.ecsv')
            
            # Grab and reformat ions
            ion_list = table['Ion']
            split_ions = [ion_name.split() for ion_name in ion_list]

            # Convert ion names to CHIANTI format
            reformatted_ions = []
            for element, state in split_ions:
                if ']' in state:
                    ion_state = str(roman.fromRoman(state.rstrip(']')))
                else:
                    ion_state = str(roman.fromRoman(state))
                reformatted_ions.append(f'{element.lower()}_{ion_state}')

            # Add new names to table
            table['Ion'] = reformatted_ions
            return table
            
        except Exception as e:
            raise ValueError(f"Failed to create CHIANTI table: {str(e)}")

    def _process_ion(self, 
                     ion_str: str, 
                     density: np.ndarray) -> Optional[object]:
        """
            Initializes a CHIANTI ion object and calculates its emissivity.
            Arguments:
                ion_str (str): Ion name in CHIANTI format (e.g., 'c_3')
                density (ndarray): Electron density array corresponding to temperature array
            Returns:
                object: Initialized CHIANTI ion object or None if initialization fails
        """
        try:
            ion = ch.ion(ion_str, 
                         temperature=self.temp, 
                         eDensity=density, 
                         abundance=self.abundance_file)
            ion.intensity()
            return ion
        
        except (AttributeError, KeyError) as e:
            print(f"Warning: Failed to process ion {ion_str}: {str(e)}")
            return None

    def _gtmat_regular_ions(self, 
                            ion_str: str, 
                            density: np.ndarray) -> None:
        """
            Processes special ions (H and He) for two-photon and continuum emissions.
            These ions require special handling beyond the normal line emission calculation.
            Arguments:
                ion_str (str): Ion name in CHIANTI format (e.g., 'h_1', 'he_2')
                density (ndarray): Electron density array corresponding to temperature array
            Returns:
                None
        """
        # Initialize the ion
        curr_ion = self._process_ion(ion_str, density)
        if curr_ion is None:
            print(f"Skipping {ion_str} due to initialization failure")
            return

        # Skip two-photon calculation for these specific ions
        if ion_str not in ['h_2', 'he_3']:
            try:
                curr_ion.twoPhoton(self.wave_arr)

                if 'intensity' in curr_ion.TwoPhoton.keys():
                    # Add two-photon contribution
                    twophot_contrib = curr_ion.TwoPhoton['intensity'].T
                    twophot_contrib *= self.bin_arr.reshape((len(self.bin_arr), 1))
                    tp_mask = np.where(np.isfinite(twophot_contrib))
                    self.gtmat[tp_mask] += twophot_contrib[tp_mask]

                else:
                    print(f'No two-photon intensity calculated for {ion_str}')

            except Exception as e:
                print(f'Two-photon calculation failed for {ion_str}: {str(e)}')

        # Process continuum contributions for these specific ions
        if ion_str in ['h_2', 'he_2', 'he_3']:
            try:
                # Initialize continuum processes
                cont = ch.continuum(ion_str, 
                                    self.temp, 
                                    abundance=self.abundance_file)
                
                # Calculate free-free emission
                self._add_freefree_contribution(ion_str, cont)

                # Calculate free-bound emission
                self._add_freebound_contribution(ion_str, cont)
                    
            except Exception as e:
                print(f'Continuum calculation failed for {ion_str}: {str(e)}')
                
    def _add_freefree_contribution(self, 
                                   ion_str: str, 
                                   cont: object) -> None:
        """
            Calculates and adds free-free (Bremsstrahlung) emission contribution to
            the G(T) matrix for a given ion.
            Arguments:
                ion_str (str): Ion name in CHIANTI format
                cont (object): CHIANTI continuum object
            Returns:
                None
        """
        try:
            # Calculate the free-free emission across the wavelength array
            cont.freeFree(self.wave_arr)

            # Check if the result contains 'intensity' data
            if 'intensity' in cont.FreeFree.keys():
                # Transpose the intensity array to align with expected dimensions
                freefree_contrib = cont.FreeFree['intensity'].T

                # Apply the binning factor across wavelength bins
                freefree_contrib *= self.bin_arr.reshape((len(self.bin_arr), 1))

                # Mask out any non-finite values 
                ff_mask = np.where(np.isfinite(freefree_contrib))

                 # Add the valid ff contributions to gtmat
                self.gtmat[ff_mask] += freefree_contrib[ff_mask]
            else:
                print(f'No FreeFree intensity calculated for {ion_str}')

        except Exception as e:
            print(f'FreeFree calculation failed for {ion_str}: {str(e)}')

    def _update_ion_list_data(self, ion_str):
        """
            Updates the flux and error values for a given ion based on
            observed lines in the CHIANTI table.
            Arguments:
                ion_str (str): Ion name in CHIANTI format
            Returns:
                None  
        """
        # Track the current ion's fluxes
        ion_mask = np.where(self.chianti_table['Ion'] == ion_str)

        # Update flux and error if the ion has observed lines
        if len(ion_mask[0]) > 0:
            ion_idx = np.where(self.ion_list == ion_str)[0][0]
            self.ion_fluxes[ion_idx] = np.sum(self.chianti_table['Flux'][ion_mask].data)
            self.ion_errs[ion_idx] = np.sqrt(
                            np.sum((self.chianti_table['Error'][ion_mask].data)**2))

    def _gtmat_single_ion(self, 
                          ion_str: str, 
                          wavelength_lower: np.ndarray, 
                          wavelength_upper: np.ndarray, 
                          density: np.ndarray) -> None:
        """
            Calculates and adds the emission line contributions for a single ion
            to the G(T) matrix. This is the main function for building the G(T) matrix.
            Arguments:
                ion_str (str): Ion name in CHIANTI format
                wavelength_lower (ndarray): Lower bounds of wavelength bins
                wavelength_upper (ndarray): Upper bounds of wavelength bins
                density (ndarray): Electron density array
            Returns:
                None
        """
        # Initialize the ion
        curr_ion = self._process_ion(ion_str, density)
        if curr_ion is None:
            return

        # Create prefactor for G(T) calculation
        gtmat_prefactor = (curr_ion.Abundance * curr_ion.IoneqOne) / curr_ion.EDensity
        
        # Scale by abundance for elements heavier than He (Z > 2)
        if curr_ion.Z > 2.0:
            gtmat_prefactor *= 10.0**self.abundance

        # Update flux and error if the ion has observed lines
        self._update_ion_list_data(ion_str)
            
        # Iterate through wavelength pairs and add contributions
        try:
            wavelengths = zip(wavelength_lower, wavelength_upper)
            for i, (low, high) in enumerate(wavelengths):
                # Create a mask for wavelengths in this bin
                mask = (curr_ion.Emiss['wvl'] > low) & (curr_ion.Emiss['wvl'] <= high)
                
                # Add contributions from each line in this wavelength bin
                for line_idx in np.where(mask)[0]:
                    self.gtmat[i, :] += gtmat_prefactor * curr_ion.Emiss['emiss'][line_idx]

        except Exception as e:
            print(f'Failed to process wavelength contributions for {ion_str}: {str(e)}')

    def _update_ion_flux_data(self, ion_str):
        """
            Updates the flux and error values for a given ion when loading
            an existing G(T) matrix.
            Arguments:
                ion_str (str): Ion name in CHIANTI format
            Returns:
                None
        """
        # Find measurements for this ion
        ion_mask = np.where(self.chianti_table['Ion'] == ion_str)[0]
        
        # Skip if no measurements
        if len(ion_mask) == 0:
            return
            
        # Get ion index
        try:
            ion_idx = np.where(self.ion_list == ion_str)[0][0]
        except IndexError:
            print(f"Warning: Ion {ion_str} not found in ion_list")
            return
        
        # Get flux values
        fluxes = self.chianti_table['Flux'][ion_mask].data
        self.ion_fluxes[ion_idx] = np.sum(fluxes)
        
        errors = self.chianti_table['Error'][ion_mask].data
        self.ion_errs[ion_idx] = np.sqrt(np.sum(errors**2))

    # ------------------------------
    # Public Methods
    # ------------------------------
    def initialize(self) -> None:
        """
            Sets up the initial parameters for G(T) matrix calculation including
            temperature array, wavelength arrays, and ion lists. Must be called
            before generating the G(T) matrix.
            Arguments:
                None
            Returns:
                None
        """
        # Create temperature array
        self.temp = np.logspace(self.min_templog, self.max_templog, self.npoints)
        
        # Generate wavelength arrays
        self.wave_arr, self.bin_arr = self._gen_rconst_arr()
        
        # Define regular ions list
        self.ions = masterListRead()
        self.reg_ions = ['h_1', 'h_2', 'he_1', 'he_2', 'he_3']
        
        # Generate GTMatrix filename identifier
        self.gtmat_str = (f'gtmat_{self.star_name}'
                          f'_w{str(self.min_wavelength)}'
                          f'_w{str(self.max_wavelength)}'
                          f'_t{str(self.min_templog)}'
                          f'_t{str(self.max_templog)}'
                          f'_r{str(self.rconst)}')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.gtmat_dir):
            os.makedirs(self.gtmat_dir)
            
        print(f"Initialized GTMatrix with {self.npoints} temperature points and {len(self.wave_arr)} wavelength points")

    def load_line_data(self) -> None:
        """
            Loads and processes emission line data from the ECSV file.
            Converts ion names to CHIANTI format and extracts unique ions,
            fluxes, and errors.
            Arguments:
                None
            Returns:
                None
        """
        # Create the CHIANTI table
        self.chianti_table = self._create_chianti_table()

        # Filter out lines with negative flux or zero error (noise)
        valid_lines = np.where(self.chianti_table['Error'] != 0.0)[0]

        # Create a new filtered table
        self.chianti_table = self.chianti_table[valid_lines]
        
        # Extract unique ions and initialize arrays
        self.ion_list = np.unique(self.chianti_table['Ion'])
        self.ion_fluxes = np.zeros(len(self.ion_list))
        self.ion_errs = np.zeros_like(self.ion_fluxes)

        print(f"Loaded emission line data: {len(self.ion_list)} unique ions with emission line measurements")

    def generate_gtmatrix(self, 
                          pressure: float = None) -> None: # UPDATE ME TO BE ABLE TO PROCESS MULTIPLE PRESSURE LISTS ???
        """
            Generates the G(T) matrix for a given pressure or loads an existing one.
            This is the main function that calculates the temperature-dependent
            contribution functions for all ions and wavelengths.
            Arguments:
                pressure (float, optional): Pressure value in dyne/cm². If None,
                                          uses the first pressure in pressure_list.
            Returns:
                None
        """
        # Use default pressure if none provided
        if pressure is None:
            pressure = self.pressure_list[0]
            
        # Calculate density from pressure and temperature
        density = pressure / self.temp
            
        # Create upper and lower bounds for the wavelengths
        wavelength_lower = self.wave_arr - (0.5 * self.bin_arr)
        wavelength_upper = wavelength_lower + self.bin_arr

        # Create the pressure-specific filename
        curr_gtmat_str = (f'{self.gtmat_str}_p{str(int(np.log10(pressure)))}'
                          f'_{self.abundance_type}.npy')
        curr_gtmat_path = f'{self.gtmat_dir}/{curr_gtmat_str}'
        
        # Check if matrix already exists
        if exists(curr_gtmat_path):
            self.gtmat = np.load(curr_gtmat_path)

            # Update flux data for each ion still needed
            for ion in self.ion_list:
                self._update_ion_flux_data(ion)

            print(f'Loaded existing matrix: {curr_gtmat_str}')
            return
            
        # Initialize with zeros for full matrix
        self.gtmat = np.zeros((len(wavelength_lower), len(self.temp)))
        
        print(f'Generating: {curr_gtmat_str}')
        
        # Process each ion and add its contribution
        for ion in tqdm(self.ions, desc="Processing main ions"):
            self._gtmat_single_ion(ion, wavelength_lower, wavelength_upper, density)
        
        # Process 'regular' ions for continuum processes
        for ion in tqdm(self.reg_ions, desc="Processing H/He ions"):
            self._gtmat_regular_ions(ion, density)
            
        # Save the generated matrix
        np.save(curr_gtmat_path, self.gtmat)
        print(f"G(T) matrix saved to {curr_gtmat_path}")

    def generate_all_matrices(self) -> None:
        """
            Generates G(T) matrices for all pressures in the pressure_list.
            A convenient wrapper that calls generate_gtmatrix for each pressure.
            Arguments:
                None
            Returns:
                None
        """
        # Process each pressure in the list
        for pressure in self.pressure_list:
            self.generate_gtmatrix(pressure)

    def generate_heatmap(self) -> None:
        """
            Creates a heatmap visualization of the G(T) matrix showing
            the temperature and wavelength dependence of emission.
            Saves the plot to the plots directory.
            Arguments:
                None
            Returns:
                None
            Raises:
                AttributeError: If G(T) matrix has not been generated or loaded
        """
        # Check that GTMatrix is available
        if self.gtmat is None:
            raise AttributeError("No G(T) matrix found. Run generate_gtmatrix() first.")

        # Create wavelength array for plotting
        n_wavelengths = self.gtmat.shape[0]
        wave_arr_subset = np.linspace(self.min_wavelength, self.max_wavelength, n_wavelengths)

        # Calculate reasonable vmin and vmax based on non-zero values
        non_zero_mask = self.gtmat > 0
        if non_zero_mask.any():
            vmin = np.log10(self.gtmat[non_zero_mask].min())
            vmax = np.log10(self.gtmat[non_zero_mask].max())
        else:
            vmin, vmax = -20, 0

        # Take log of the matrix, handling zeros
        log_gtmat = np.zeros_like(self.gtmat)
        log_gtmat[non_zero_mask] = np.log10(self.gtmat[non_zero_mask])
        log_gtmat[~non_zero_mask] = vmin

        # Create the plot
        fig = plt.figure(figsize=(10, 6))
        plt.pcolormesh(np.log10(self.temp), 
                       wave_arr_subset,
                       log_gtmat,
                       cmap='inferno', 
                       vmin=vmin, 
                       vmax=vmax,
                       shading='nearest')
        plt.colorbar(label=r'$\log_{10}$ G(T)')
        plt.xlabel(r'$\log_{10} (T \, \, [\mathrm{K}]) $')
        plt.ylabel(r'Wavelength [$\mathrm{\AA}$]')
        plt.title(f'{self.star_name} G(T) Matrix ({self.abundance_type})')
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
        
        # Save if a path is provided
        save_path = f'plots/gtmat_{self.star_name}_heatmap.png'
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")

    @staticmethod
    def calculate_integral(gtmat: np.ndarray, 
                           temp: np.ndarray, 
                           psi_model: np.ndarray, 
                           flux_weighting: float) -> np.ndarray:
        """
            Calculates the integral of G(T) * psi(T) over temperature for all wavelengths.
            This is used to compute the predicted emission spectrum from a DEM model.
            Arguments:
                gtmat (ndarray): G(T) matrix with shape (n_wavelengths, n_temperatures)
                temp (ndarray): Temperature array
                psi_model (ndarray): Differential Emission Measure model
                flux_weighting (float): Scaling factor to convert to observed flux
            Returns:
                ndarray: Integrated emission intensity for each wavelength
        """
        # Multiply G(T) by psi(T) for all wavelengths at once
        integrand = gtmat * psi_model[np.newaxis, :]
        
        # Integrate over temperature dimension for all wavelengths at once
        intensity_array = np.trapz(integrand, temp, axis=1) * flux_weighting
        
        return intensity_array
        
    def get_emission_line_indices(self):
        """
            Finds the indices in the wavelength array that correspond to
            observed emission lines. For each ion, uses the strongest line.
            Arguments:
                None
            Returns:
                ndarray: Array of wavelength indices for observed emission lines
            Raises:
                ValueError: If no emission line data has been loaded
        """
        if not hasattr(self, 'ion_list') or self.ion_list is None:
            raise ValueError("No emission line data loaded. Call load_line_data() first.")
        
        # Initialize array to store indices
        line_indices = []
        
        # Process each unique ion
        for _, ion in enumerate(self.ion_list):
            # Find all measurements for this ion in the CHIANTI table
            ion_mask = np.where(self.chianti_table['Ion'] == ion)[0]
            
            if len(ion_mask) > 0:
                print(ion)
                # Get wavelengths for this ion
                wvls = self.chianti_table['Rest Wavelength'][ion_mask].data # APPLY LOGIC EVERYWHERE !
                fluxes = self.chianti_table['Flux'][ion_mask].data
                
                # Use strongest line for each ion
                strongest_idx = np.argmax(fluxes)
                wavelength = wvls[strongest_idx]
                
                # Find closest wavelength bin
                idx = np.argmin(np.abs(self.wave_arr - wavelength))
                line_indices.append(idx)
        
        # Store indices for later use
        self.emission_line_indices = np.array(line_indices)
        
        return self.emission_line_indices