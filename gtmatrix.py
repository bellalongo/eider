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


# Set plot styles
sns.set_context('paper')
sns.set_style('ticks')
plt.rc('font', family='serif')
plt.rc('text', usetex=True)


class GTMatrix:
    def __init__(self, 
                 star_name: str, 
                 abundance: float, 
                 abundance_file: str, 
                 flux_file: str, 
                 gtmat_dir: str):
        """
            Initialize the GTMatrix class with a directory for saving G(T) matrices.
        """
        self.star_name = star_name
        self.abundance = abundance
        self.abundance_file = abundance_file
        self.flux_file = flux_file
        self.gtmat_dir = gtmat_dir

        # Variable definitions
        self.min_wavelength = 1 # Minimum wavelength to create GT matrix
        self.max_wavelength = 1500 # Maximum wavelength to create GT matrix

        self.rconst = 100 # Spectral resolving power
        self.min_templog = 4 # Minimum temperature in log scale
        self.max_templog = 8 # Maximum temperature in log scale
        self.npoints = 200

        # Set abundance type
        self.abundance_type = self.get_abund_type()

        # Define desired pressure list
        self.pressure_list = [1e17]

        # Grab ions from Chianti
        self.ions = masterListRead()
        self.reg_ions = ['h_1', 'h_2', 'he_1', 'he_2', 'he_3']

        # Grab Chianti table and ion data
        self.chianti_table = self.create_chianti_table()
        self.ion_list = np.unique(self.chianti_table['Ion'])
        self.ion_fluxes = np.zeros(len(self.ion_list)) # Will be populated later
        self.ion_errs = np.zeros_like(self.ion_fluxes) # Will be populated later

        # Ensure the directory exists
        if not os.path.exists(gtmat_dir):
            os.makedirs(gtmat_dir)

        # Generate string to where the gtmat will be stored
        self.gtmat_str = f'gtmat_{star_name
                                  }_w{str(self.min_wavelength)
                                      }_w{str(self.max_wavelength)
                                          }_t{str(self.min_templog)
                                              }_t{str(self.max_templog)
                                                  }_r{str(self.rconst)}'

        # Create temperature distribution
        self.temp = np.logspace(self.min_templog, self.max_templog, self.npoints)

        # Generate rconst arr
        self.wave_arr, self.bin_arr = self.gen_rconst_arr()

        # Generate gtmat
        self.gen_gmat_press_library()

        # Visualize 
        self.gtmat_heatmap()


    def create_chianti_table(self) -> Table: 
        """

        """
        # Read from the ecsv file
        table = Table.read(f'lines/{self.flux_file}', format='ascii.ecsv')
        
        # Grab and reformat ions
        ion_list = table['Ion']
        split_ions = [ion_name.split() for ion_name in ion_list]

        reformatted_ions = [
            f'{element.lower()}_{str(roman.fromRoman(state.rstrip(']') if ']' in state else state))}'
            for element, state in split_ions
        ]

        # Add new names to table
        table['Ion'] = reformatted_ions

        return table


    def get_abund_type(self) -> str:
        """

        """
        if self.abundance == 0.0:
            return 'sol0'
        elif self.abundance == -1.0:
            return 'sub1'
        elif self.abundance == 1.0:
            return 'sup1'
        

    def gen_rconst_arr(self):
        """

        """
        # Create a wavelength and bin array
        wave_arr, bin_arr = [], []

        for _, temp_wave in enumerate(np.arange(self.min_wavelength, self.max_wavelength, self.min_wavelength / self.rconst)):
            temp_dlambda = temp_wave / self.rconst
            bin_arr.append(temp_dlambda)
            wave_arr.append(temp_wave + (0.5 * temp_dlambda))

        return np.array(wave_arr), np.array(bin_arr)
    
    
    def process_ion(self, 
                    ion_str: str, 
                    density: float):
        """

        """
        try:
            ion = ch.ion(ion_str, temperature=self.temp, 
                        eDensity=density, 
                        abundance=self.abundance_file)
            ion.intensity()
            return ion
        except (AttributeError, KeyError) as e:
            print(f"Warning: Failed to process ion {ion_str}: {str(e)}")
            return None


    def gtmat_regular_ions(self, 
                           ion_str: str, 
                           density: float):
        """

        """
        # Initialize the ion
        curr_ion = self.process_ion(ion_str, density)
        if curr_ion is None:
            print(f"Skipping {ion_str} due to initialization failure")
            return

        # Skip two-photon calculation for these specific ions
        if ion_str in ['h_2', 'he_3']:
            pass
        else:
            try:
                curr_ion.twoPhoton(self.wave_arr)
                if 'intensity' in curr_ion.TwoPhoton.keys():
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
                cont = ch.continuum(ion_str, self.temp, 
                                abundance=self.abundance_file)
                
                # Calculate free-free emission
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

                # Calculate free-bound emission
                try:
                    cont.freeBound(self.wave_arr)
                    if 'intensity' in cont.FreeBound.keys():
                        freebound_contrib = cont.FreeBound['intensity'].T
                        freebound_contrib *= self.bin_arr.reshape((len(self.bin_arr), 1))
                        fb_mask = np.where(np.isfinite(freebound_contrib))
                        self.gtmat[fb_mask] += freebound_contrib[fb_mask]
                    else:
                        print(f'No FreeBound intensity calculated for {ion_str}')
                except Exception as e:
                    print(f'FreeBound calculation failed for {ion_str}: {str(e)}')
                    
            except Exception as e:
                print(f'Continuum calculation failed for {ion_str}: {str(e)}')

    def gtmat_single_ion(self, 
                         ion_str, 
                         wavelength_lower, 
                         wavelength_upper, 
                         density):
        """

        """
        # Initialize the ion
        curr_ion = self.process_ion(ion_str, density)
        if curr_ion is None:
            return

        # Create prefactor
        gtmat_prefactor = (curr_ion.Abundance * curr_ion.IoneqOne) / curr_ion.EDensity
        gtmat_prefactor = gtmat_prefactor * (10.0**self.abundance) if curr_ion.Z > 2.0 else gtmat_prefactor

        # Track the current ion's fluxes
        ion_mask = np.where(self.chianti_table['Ion'] == ion_str)

        # Check if the ion has observed lines
        if len(ion_mask[0]) > 0:
            ion_idx = np.where(self.ion_list == ion_str)[0][0]
            self.ion_fluxes[ion_idx] = np.sum(self.chianti_table['Flux'][ion_mask])
            self.ion_errs[ion_idx] = np.sqrt(
                np.sum((self.chianti_table['Error'][ion_mask])**2))

        # Iterate through wavelength pairs
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


    def gen_gtmatrix(self, wavelength_lower, wavelength_upper, density):
        """

        """
        # Initialize with zeros for full matrix
        self.gtmat = np.zeros((len(wavelength_lower), len(self.temp)))
        
        # Process each ion and add its contribution
        for ion in tqdm(self.ions, desc="Processing main ions"):
            self.gtmat_single_ion(ion, wavelength_lower, wavelength_upper, density)
        
        # Process 'regular' ions
        for ion in tqdm(self.reg_ions, desc="Processing H/He ions"):
            self.gtmat_regular_ions(ion, density)

        return self.gtmat


    def gen_gmat_press_library(self):
        """

        """
        # Create upper and lower bounds for the wavelengths
        wavelength_lower = self.wave_arr - (0.5*self.bin_arr)
        wavelength_upper = wavelength_lower + self.bin_arr

        for pressure in self.pressure_list:
            density = pressure / self.temp

            # Create a temporary string
            curr_gtmat_str = f'{self.gtmat_str}_p{str(int(np.log10(pressure)))}_{self.abundance_type}.npy'
            curr_gtmat_path = f'{self.gtmat_dir}/{curr_gtmat_str}'
            
            # Check if exists
            if exists(curr_gtmat_path):
                self.gtmat = np.load(curr_gtmat_path)
                continue

            # Print(ions)
            print(f'Generating: {curr_gtmat_str}')

            # Generate GTMatrix
            self.gtmat = self.gen_gtmatrix(wavelength_lower, wavelength_upper, density)

            np.save(f'{self.gtmat_dir}/{curr_gtmat_str}', self.gtmat)


    def gtmat_heatmap(self):
        """

        """
        # Check that GTMatrix was saved
        if not hasattr(self, 'gtmat'):
            raise AttributeError("No G(T) matrix found. Run gen_gmat_press_library first.")

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
        plt.figure()
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
        plt.tight_layout()

        # Save and show the plot
        plt.savefig(f'{self.gtmat_dir}/gtmat_heatmap_{self.star_name}.pdf', dpi=5, bbox_inches='tight')
        plt.show()


    @staticmethod
    def gtmat_integral(gtmat: np.ndarray,
                       temp: np.nparray,
                       psi_model: np.ndarray,
                       flux_weighting: float) -> np.ndarray:
        """

        """
        # Create the array
        intensity_array = [np.trapz(
            gtmat[i, :] * psi_model, 
            temp) * flux_weighting 
            for i in range(len(np.shape(gtmat)[0]))] 

        return np.array(intensity_array)