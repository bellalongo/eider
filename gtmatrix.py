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

    """
    
    def __init__(self, 
                 star_config, 
                 gtmat_config, 
                 path_config):
        """

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

        """
        # Extract star parameters from star config
        self.star_name = self.star_config['star_name']
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

        """
        try:
            cont.freeFree(self.wave_arr)
            if 'intensity' in cont.FreeFree.keys():
                freefree_contrib = cont.FreeFree['intensity'].T
                freefree_contrib *= self.bin_arr.reshape((len(self.bin_arr), 1))
                ff_mask = np.where(np.isfinite(freefree_contrib))
                self.gtmat[ff_mask] += freefree_contrib[ff_mask]
            else:
                print(f'No FreeFree intensity calculated for {ion_str}')
        except Exception as e:
            print(f'FreeFree calculation failed for {ion_str}: {str(e)}')

    def _gtmat_single_ion(self, 
                          ion_str: str, 
                          wavelength_lower: np.ndarray, 
                          wavelength_upper: np.ndarray, 
                          density: np.ndarray) -> None:
        """

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

        # Track the current ion's fluxes
        ion_mask = np.where(self.chianti_table['Ion'] == ion_str)

        # Update flux and error if the ion has observed lines
        if len(ion_mask[0]) > 0:
            ion_idx = np.where(self.ion_list == ion_str)[0][0]
            self.ion_fluxes[ion_idx] = np.sum(self.chianti_table['Flux'][ion_mask])
            self.ion_errs[ion_idx] = np.sqrt(
                np.sum((self.chianti_table['Error'][ion_mask])**2))

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

    # ------------------------------
    # Public Methods
    # ------------------------------
    def initialize(self) -> None:
        """

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
                          pressure: float = None) -> None:
        """

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
            print(f'Loaded existing matrix: {curr_gtmat_str}')
            return
            
        # Make sure line data is loaded
        if self.chianti_table is None:
            self.load_line_data()
            
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

        """
        # Make sure we're initialized
        if self.temp is None or self.wave_arr is None:
            self.initialize()
            
        # Process each pressure in the list
        for pressure in self.pressure_list:
            self.generate_gtmatrix(pressure)

    def generate_heatmap(self) -> None:
        """

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
        save_path = f'plots/{self.star_name}_gtmat_heatmap'
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")

    @staticmethod
    def calculate_integral(gtmat: np.ndarray, 
                           temp: np.ndarray, 
                           psi_model: np.ndarray, 
                           flux_weighting: float) -> np.ndarray:
        """

        """
        # Multiply G(T) by psi(T) for all wavelengths at once
        integrand = gtmat * psi_model[np.newaxis, :]
        
        # Integrate over temperature dimension for all wavelengths at once
        intensity_array = np.trapz(integrand, temp, axis=1) * flux_weighting
        
        return intensity_array
        
    def get_emission_line_indices(self):
        """

        """
        if not hasattr(self, 'ion_list') or self.ion_list is None:
            raise ValueError("No emission line data loaded. Call load_line_data() first.")
            
        # Get wavelengths of emission lines from the chianti_table
        emission_line_wavelengths = self.chianti_table['Rest Wavelength']
        
        # Initialize array to store indices
        line_indices = []

        # Process each unique ion
        for ion in self.ion_list:
            # Find all measurements for this ion
            ion_mask = self.chianti_table['Ion'] == ion
            
            # Get wavelengths and fluxes for this ion
            ion_wavelengths = self.chianti_table['Rest Wavelength'][ion_mask]
            ion_fluxes = self.chianti_table['Flux'][ion_mask]
            
            # Choose the representative wavelength (strongest line)
            if len(ion_wavelengths) > 0:
                # Use the wavelength with the highest flux
                best_idx = np.argmax(ion_fluxes)
                rep_wavelength = ion_wavelengths[best_idx]
                
                # Find the corresponding index in the wavelength grid
                index = np.argmin(np.abs(self.wave_arr - rep_wavelength))
                
                # Make sure the index is within bounds
                if 0 <= index < len(self.wave_arr):
                    line_indices.append(index)
        
        # Save these indices for later use
        self.emission_line_indices = np.array(line_indices)

        # Double-check that the number of indices matches the number of ions
        if len(line_indices) != len(self.ion_list):
            print(f"Warning: Number of indices ({len(line_indices)}) doesn't match number of ions ({len(self.ion_list)})")

        # Print diagnostic information
        print(f"Found {len(line_indices)} emission line indices")
        
        return self.emission_line_indices