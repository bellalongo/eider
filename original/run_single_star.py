import corner
import resample
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table
from data_prep import generate_constant_R_wave_arr
from data_prep import generate_spectrum_from_samples
from gofnt_routines import parse_ascii_table_CHIANTI, resample_gofnt_matrix
from gofnt_routines import generate_ion_gofnts
from fitting import fit_emcee, ln_likelihood_dem
from astropy.io import fits
from dem_plots import plot_dem, plot_spectrum, display_fig, compare_ion, compare_spec
from tinygp import kernels, GaussianProcess
from gofnt_routines import do_gofnt_matrix_integral
from jax import numpy as jnp
from multiprocessing import Pool


def generate_flux_weighting(star_name, star_dist, star_rad):
    flux_weighting = ((np.pi * u.sr * (star_rad**2.0) *
                       (1.0 / (star_dist**2.0))).to(u.sr)).value
    np.save('flux_weighting_' + star_name, [flux_weighting])
    return flux_weighting


def get_best_gofnt_matrix_press(abundance, press, abund_type='sol0',
                                mode='r100'):
    gofnt_dir = 'gofnt_dir/'
    gofnt_root = 'gofnt_w1_w1500_t4_t8_' + mode + '_p'
    gofnt_matrix = np.load(gofnt_dir + gofnt_root
                           + str(int(np.log10(press)))
                           + '_' + abund_type + '.npy')
    gofnt_matrix *= 10.0**abundance
    return gofnt_matrix


def get_line_data_gofnts(star_name, line_table_file, abundance,
                         temp, dens, bin_width):
    line_table = parse_ascii_table_CHIANTI(line_table_file)
    gofnt_lines, flux, err, names = generate_ion_gofnts(line_table,
                                                        abundance,
                                                        bin_width,
                                                        temp,
                                                        dens)
    np.save('gofnt_lines_' + star_name + '.npy', gofnt_lines)
    np.save('ion_fluxes_' + star_name + '.npy', flux)
    np.save('ion_errs_' + star_name + '.npy', err)
    np.save('ion_names_' + star_name + '.npy', names)
    return gofnt_lines, flux, err, names


def get_spectrum_data_gofnt(star_name, data_npy_file, gofnt_matrix):
    wave, wave_bins, flux, err = np.load(data_npy_file, allow_pickle=True)
    flux *= wave_bins
    err *= wave_bins
    wave_old, _ = generate_constant_R_wave_arr(1, 1500, 100)
    gofnt_spectrum = resample_gofnt_matrix(gofnt_matrix, wave, wave_bins,
                                           wave_old)
    temp = np.logspace(4, 8, 200)
    gofnt_ints = np.trapz(gofnt_spectrum, temp)
    mask = np.where(gofnt_ints >=
                    (np.max(gofnt_ints) / 1e3))
    np.save('gofnt_spectrum_' + star_name + '.npy', gofnt_spectrum[mask[0], :])
    np.save('spectrum_fluxes_' + star_name + '.npy', flux[mask])
    np.save('spectrum_errs_' + star_name + '.npy', err[mask])
    np.save('spectrum_waves_' + star_name + '.npy', wave[mask])
    np.save('spectrum_bins_' + star_name + '.npy', wave_bins[mask])
    return gofnt_spectrum, flux, err


def get_star_data_gofnt_press(star_name, abundance, press,
                              line_table_file=None, data_npy_file=None,
                              bin_width=1.0):
    big_gofnt = get_best_gofnt_matrix_press(abundance, press)
    temp = np.logspace(4, 8, 200)
    dens = press / temp
    if line_table_file is not None:
        if os.path.isfile('gofnt_lines_' + star_name + '.npy'):
            gofnt_lines = np.load('gofnt_lines_' + star_name + '.npy')
            flux = np.load('ion_fluxes_' + star_name + '.npy')
            err = np.load('ion_errs_' + star_name + '.npy')
        else:
            gofnt_lines, flux, err, _ = get_line_data_gofnts(star_name,
                                                             line_table_file,
                                                             abundance,
                                                             temp, dens,
                                                             bin_width)
        line_flux = flux
        line_err = err
    else:
        gofnt_lines = None
    if data_npy_file is not None:
        if os.path.isfile('gofnt_spectrum_' + star_name + '.npy'):
            gofnt_spectrum = np.load('gofnt_spectrum_' + star_name + '.npy')
            flux = np.load('spectrum_fluxes_' + star_name + '.npy')
            err = np.load('spectrum_errs_' + star_name + '.npy')
        else:
            gofnt_spectrum, flux, err = get_spectrum_data_gofnt(star_name,
                                                                data_npy_file,
                                                                big_gofnt)
        spectrum_flux = flux
        spectrum_err = err
    else:
        gofnt_spectrum = None
    if (gofnt_lines is None):
        if (gofnt_spectrum is None):
            print('Where is this star\'s data to do anything with?')
        else:
            gofnt_matrix = gofnt_spectrum
            np.save('gofnt_' + star_name + '.npy', gofnt_matrix)
            np.save('flux_' + star_name + '.npy', flux)
            np.save('err_' + star_name + '.npy', err)
    elif (gofnt_spectrum is None):
        gofnt_matrix = gofnt_lines
        np.save('gofnt_' + star_name + '.npy', gofnt_matrix)
        np.save('flux_' + star_name + '.npy', flux)
        np.save('err_' + star_name + '.npy', err)
    else:
        gofnt_matrix = np.append(gofnt_spectrum, gofnt_lines, axis=0)
        flux = np.append(spectrum_flux, line_flux)
        err = np.append(spectrum_err, line_err)
        np.save('gofnt_' + star_name + '.npy', gofnt_matrix)
        np.save('flux_' + star_name + '.npy', flux)
        np.save('err_' + star_name + '.npy', err)
    return gofnt_matrix, flux, err


def run_mcmc_single_star(init_pos, gofnt_matrix, flux, err, flux_weighting,
                         star_name,
                         n_walkers=50, burn_in_steps=200,
                         production_steps=800, thread_num=5,
                         count_num=2000):
    temp = np.logspace(4, 8, 200)
    log_temp = np.log10(temp)
    samples, lnprob, sampler = fit_emcee(init_pos=init_pos,
                                         likelihood_func=ln_likelihood_dem,
                                         likelihood_args=[flux,
                                                          err,
                                                          log_temp,
                                                          temp,
                                                          gofnt_matrix,
                                                          flux_weighting],
                                         n_walkers=n_walkers,
                                         burn_in_steps=burn_in_steps,
                                         production_steps=production_steps,
                                         thread_num=thread_num,
                                         count_num=count_num)
    np.save('samples_' + star_name, samples)
    np.save('lnprob_' + star_name, lnprob)
    return samples, lnprob, sampler

def generate_spectrum_data_npy(star_name):
    def read_chunks(file,
                    spec_skiprows,spec_nrows,
                    resid_skiprows,resid_nrows):
        spec_df = pd.read_csv(file,
                              skiprows=spec_skiprows, nrows=spec_nrows,
                              header=None, delim_whitespace=True,
                              names=['energy', 'energy_err',
                                     'spec','spec_err','bestfit_model',
                                     'mod1','mod2','mod3','mod4'])

        resid_df = pd.read_csv(file,
                               skiprows=resid_skiprows, nrows=resid_nrows,
                               header=None, delim_whitespace=True,
                               names=['energy', 'energy_err',
                                      'resid','resid_err',
                                      'no1','no2','no3','no4','no5'])
        return(spec_df,resid_df)
    spec, resid = read_chunks('unbinned_fit_resid.dat',
                              3, 1500, 3, 1500)
    plt.errorbar(spec['energy'], spec['spec'], label='data',
                 drawstyle='steps-mid')
    plt.plot(spec['energy'], spec['bestfit_model'], label='best')
    plt.plot(spec['energy'], spec['mod1'], label='mod1')
    plt.plot(spec['energy'], spec['mod2'], label='mod2')
    plt.legend()
    plt.plot()
    plt.show()

    xray_energy = np.array((spec['energy'])) * u.keV
    xray_energy_upps = xray_energy + (5e-3 * u.keV)
    xray_energy_lows = xray_energy - (5e-3 * u.keV)
    print(spec['energy'])

    xray_wave_lows = (xray_energy_upps).to(u.AA, equivalencies=u.spectral()) # XRAY STUFF HERE
    xray_wave_upps = (xray_energy_lows).to(u.AA, equivalencies=u.spectral())
    xray_wave2 = (xray_energy).to(u.AA, equivalencies=u.spectral())
    xray_wave = 0.5 * (xray_wave_lows + xray_wave_upps)
    xray_bins = (xray_wave_upps - xray_wave_lows)

    flux_unit_old = 1.0 / (u.s * u.cm * u.cm * u.keV)
    if star_name == 'v1298_tau_y_freefree':
        xray_flux_old = np.array(spec['spec']) * flux_unit_old
    elif star_name == 'v1298_tau_n_freefree':
        xray_flux_old = np.array(spec['spec'] - spec['mod2']) * flux_unit_old
    xray_flux_old *= xray_energy.to(u.erg)
    xray_err_old = np.array(spec['spec_err']) * flux_unit_old
    xray_err_old *= xray_energy.to(u.erg)

    flux_unit_new = flux_unit_old * u.erg * u.keV / u.AA

    xray_flux = ((xray_flux_old * 1e-2 * u.keV) / xray_bins).to(flux_unit_new)
    xray_err = ((xray_err_old * 1e-2 * u.keV) / xray_bins).to(flux_unit_new)

    print('Loaded X-ray Data')

    mask = np.where((xray_flux.value > 0.0)
                    & (xray_err.value > 0.0)
                    & (xray_flux > 1.0 * xray_err)
                    & np.isfinite(xray_flux) & (np.isfinite(xray_err))
                    & (xray_wave.value <= 150.0)
                    & (xray_wave.value >= 5.0))
    xray_flux = xray_flux[mask]
    xray_err = xray_err[mask]
    xray_wave = xray_wave[mask]
    xray_bins = xray_bins[mask]

    plt.errorbar(xray_wave, xray_flux, yerr=xray_err,
                 drawstyle='steps-mid')
    plt.show()


    np.save(star_name + '_spectrum_data.npy',
            [xray_wave.value, xray_bins.value,
             xray_flux.value, xray_err.value])
    return star_name + '_spectrum_data.npy'


if __name__ == '__main__':
    temp = np.logspace(4, 8, 200)
    log_temp = np.log10(temp)
    star_name_root = 'au_mic_q'
    title_name = r'AU Mic: Quiescent'
    abundance = -0.12
    star_rad = 0.75 * u.Rsun
    star_dist = 9.7140 * u.pc
    flux_weighting = generate_flux_weighting(star_name_root,
                                             star_dist, star_rad)
    init_pos = [22.49331207, -3.31678227, -0.49848262,
                -1.27244452, -0.93897032, -0.67235648,
                -0.08085897]
    press_list = [1e17, 1e16, 1e15, 1e13, 1e14, 1e18, 1e19, 1e20, 1e21, 1e22]
    # if os.path.isfile(star_name_root + '_spectrum_data.npy'):
    #     data_npy_file = star_name_root + '_spectrum_data.npy'
    # else:
    #     data_npy_file = generate_spectrum_data_npy(star_name_root)
    line_table_file = star_name_root + '_linetable.ascii'
    for press in press_list[:1]:
        star_name = star_name_root + '_p' + str(int(np.log10(press)))
        gofnt_all = get_best_gofnt_matrix_press(abundance, press)
        # wave_all, bin_all = generate_constant_R_wave_arr(1, 1500, 100)
        wave_all = np.arange(1, 1501, 1.0)
        bin_all = np.ones_like(wave_all)
        if os.path.isfile('gofnt_' + star_name + '.npy'):
            gofnt_matrix = np.load('gofnt_' + star_name + '.npy')
            flux = np.load('flux_' + star_name + '.npy')
            err = np.load('err_' + star_name + '.npy')
        else:
            out = get_star_data_gofnt_press(star_name,
                                            abundance,
                                            press,
                                            line_table_file,
                                            data_npy_file=None)
            gofnt_matrix, flux, err = out
        if os.path.isfile('samples_' + star_name + '.npy'):
            samples = np.load('samples_' + star_name + '.npy')
            lnprob = np.load('lnprob_' + star_name + '.npy')
        else:
            samples, lnprob, _ = run_mcmc_single_star(init_pos,
                                                      gofnt_matrix,
                                                      flux, err,
                                                      flux_weighting,
                                                      star_name)

        if os.path.isfile('dem_' + star_name + '.pdf'):
            pass
        else:
            if os.path.isfile('gofnt_lines_' + star_name + '.npy'):
                ion_names = np.load('ion_names_' + star_name + '.npy')
                ion_gofnts = np.load('gofnt_lines_' + star_name + '.npy')
                ion_fluxes = np.load('ion_fluxes_' + star_name + '.npy')
                ion_errs = np.load('ion_errs_' + star_name + '.npy')
            else:
                ion_names = None
                ion_gofnts = None
                ion_fluxes = None
                ion_errs = None
            g = plot_dem(samples, lnprob, flux, gofnt_matrix,
                         log_temp, temp, flux_weighting,
                         'b', 'cornflowerblue', 0.1, 500,
                         'MCMC Samples', 'Best-fit DEM model',
                         title_name, ion_names=ion_names,
                         ion_gofnts=ion_gofnts, ion_fluxes=ion_fluxes)
            g = display_fig(g, 'dem_' + star_name, mode='pdf')
            plt.clf()
        if os.path.isfile('corner_' + star_name + '.pdf'):
            pass
        else:
            h = corner.corner(samples[:], quantiles=[0.16, 0.5, 0.84],
                              show_titles=True, title_kwargs={"fontsize": 12},
                              plot_contours=True)
            h = display_fig(h, 'corner_' + star_name, mode='pdf')
            plt.clf()

        if os.path.isfile('spectrum_' + star_name + '.fits'):
            spectrum_table = Table.read('spectrum_' + star_name + '.fits')

        else:
            spectrum_name = 'spectrum_' + star_name
            spectrum_table, _ = generate_spectrum_from_samples(spectrum_name,
                                                               samples,
                                                               lnprob,
                                                               gofnt_all,
                                                               flux_weighting,
                                                               wave_all,
                                                               bin_all,
                                                               sample_num=1000)
        if os.path.isfile('spectrum_' + star_name + '.pdf'):
            pass
        else:
            spec_fig = plot_spectrum('spectrum_' + star_name + '.fits',
                                     title_name)
            spec_fig = display_fig(spec_fig, 'spectrum_' + star_name,
                                   mode='pdf')

        if os.path.isfile('gofnt_lines_' + star_name + '.npy'):
            ion_names = np.load('ion_names_' + star_name + '.npy')
            ion_gofnts = np.load('gofnt_lines_' + star_name + '.npy')
            ion_fluxes = np.load('ion_fluxes_' + star_name + '.npy')
            ion_errs = np.load('ion_errs_' + star_name + '.npy')
            dems = np.load('spectrum_' + star_name + '_dems.npy')
            if os.path.isfile('compare_ion_' + star_name + '.pdf'):
                pass
            else:
                compare_ion_fig = compare_ion(ion_gofnts, ion_names, ion_fluxes, ion_errs,
                                              dems, samples, flux_weighting, temp, title_name)
                compare_ion_fig = display_fig(compare_ion_fig, 'compare_ion_' + star_name,
                                              mode='pdf')
        else:
            pass
        if os.path.isfile('gofnt_spectrum_' + star_name + '.npy'):
            spec_gofnt = np.load('gofnt_spectrum_' + star_name + '.npy')
            spec_out = np.load(star_name_root + '_spectrum_data.npy')
            spec_wave, spec_bins, spec_flux, spec_err = spec_out        
            if os.path.isfile('compare_spec_' + star_name + '.pdf'):
                pass
            else:
                compare_spec_fig = compare_spec(spec_gofnt, spec_wave, spec_bins, spec_flux, spec_err,
                                                samples, flux_weighting, temp, title_name)
                compare_spec_fig = display_fig(compare_spec_fig, 'compare_spec_' + star_name,
                                               mode='pdf')
