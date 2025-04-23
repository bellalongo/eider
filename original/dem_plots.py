import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import roman
from typing import List, Callable, Any, Union, Tuple
from numpy.polynomial.chebyshev import chebval
from scipy.integrate import cumulative_trapezoid
from astropy.io import fits
from tinygp import kernels, GaussianProcess
from jax import numpy as jnp
from gofnt_routines import do_gofnt_matrix_integral


def plot_dem(samples, lnprob, flux_arr, gofnt_matrix,
             log_temp, temp, flux_weighting,
             main_color, sample_color, alpha,
             sample_num, sample_label,
             main_label, title_name, figure=None,
             low_y=19.0, high_y=26.0, ion_names=None,
             ion_gofnts=None, ion_fluxes=None,
             dem_method='cheby', knot_locs=None):
    if figure is not None:
        plt.figure(figure.number)
    shift_log_temp = log_temp - np.mean(log_temp)
    range_temp = (np.max(shift_log_temp) - np.min(shift_log_temp))
    shift_log_temp /= (0.5 * range_temp)
    if dem_method == 'cheby':
        psi_model = 10.0**chebval(shift_log_temp, samples[np.argmax(lnprob)][:-1])
    elif dem_method == 'gp':
        params = samples[np.argmax(lnprob)]
        amp1 = np.exp(params[-3])
        amp2 = np.exp(params[-2])
        scale = params[-1]
        knots = params[:-3]
        k1 = amp2 * kernels.Matern32(scale) + kernels.Constant(amp1)
        gp = GaussianProcess(k1, jnp.array(knot_locs))
        x_test = jnp.linspace(-1, 1, len(temp))
        mu = gp.predict(knots, x_test)
        psi_model = 10.0**np.array(mu)
    total_samples = np.random.choice(len(samples), sample_num)
    psi_ys = flux_arr / (flux_weighting * np.trapz(gofnt_matrix, temp))
    temp_lows = np.min(temp) * np.ones((len(gofnt_matrix)))
    temp_upps = np.max(temp) * np.ones_like(temp_lows)
    temp_lows = 1e4 * np.ones_like(psi_ys)
    temp_upps = 1e8 * np.ones_like(temp_lows)
    for i in range(len(flux_arr)):
        gofnt_cumtrapz = cumulative_trapezoid(gofnt_matrix[i], temp)
        low_index = np.argmin(
            np.abs(gofnt_cumtrapz - (0.16 * gofnt_cumtrapz[-1])))
        upp_index = np.argmin(
            np.abs(gofnt_cumtrapz - (0.84 * gofnt_cumtrapz[-1])))
        temp_lows[i] = temp[low_index + 1]
        temp_upps[i] = temp[upp_index + 1]
    for i in range(0, sample_num):
        s = samples[total_samples[i]]
        if dem_method == 'cheby':
            temp_psi = 10.0**chebval(shift_log_temp, s[:-1])
        elif dem_method == 'gp':
            s = samples[total_samples[i]]
            amp1 = np.exp(s[-3])
            amp2 = np.exp(s[-2])
            scale = s[-1]
            knots = s[:-3]
            k1 = amp2 * kernels.Matern32(scale) + kernels.Constant(amp1)
            gp = GaussianProcess(k1, jnp.array(knot_locs))
            x_test = jnp.linspace(-1, 1, len(temp))
            mu = gp.predict(knots, x_test)
            temp_psi = 10.0**np.array(mu)
        if i == 0:
            plt.loglog(temp, temp_psi,
                       color=sample_color, alpha=alpha, label=sample_label)
        else:
            plt.loglog(temp, temp_psi, color=sample_color, alpha=alpha)
    plt.loglog(temp, psi_model, color=main_color, label=main_label)
    plt.hlines(psi_ys, temp_lows, temp_upps, label='Flux Constraints',
               colors='k', zorder=100)
    if ion_names is not None:
        dem_xs = np.array([temp[np.argmax(ion_gofnts[i])] for i in range(len(ion_names))])
        dem_ys = ion_fluxes
        dem_ys /= do_gofnt_matrix_integral(np.ones_like(temp),
                                            ion_gofnts, temp,
                                            flux_weighting)
        for i in range(len(ion_names)):
            ion_name = ion_names[i].split('_')
            new_name = ion_name[0].capitalize() + ' '
            new_name += roman.toRoman(int(ion_name[1]))
            plt.text(dem_xs[i], dem_ys[i], new_name)
    plt.ylim(10.0**low_y, 10.0**high_y)
    plt.xlabel('Temperature [K]')
    y_label = r'$\Psi(T) = N_e N_{\mathrm{H}} \frac{ds}{dT}$ '
    y_label += r'[cm$^{-5}$ K$^{-1}$]'
    plt.ylabel(y_label)
    plt.title(title_name)
    return plt.gcf()


def plot_spectrum(spec_fits, title_name,
                  figure=None, alpha=0.3, color='b'):
    hdu = fits.open(spec_fits)
    wave = hdu[1].data['Wavelength']
    flux = hdu[1].data['Flux_density']
    upp = flux + hdu[1].data['Upper_Error_84']
    low = flux - hdu[1].data['Lower_Error_16']
    plt.semilogy(wave, flux, drawstyle='steps-mid', color=color)
    plt.fill_between(wave, low, upp, color=color, alpha=alpha, step='mid')
    plt.title(title_name)
    plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
    plt.ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]')
    return plt.gcf()


def display_fig(figure, name, mode='pdf', dpi=500, legend=True):
    plt.figure(figure.number)
    if legend is True:
        plt.legend()
    plt.tight_layout()
    if mode == 'SHOW':
        plt.show()
    else:
        plt.savefig(name + '.' + mode, dpi=dpi)
        plt.clf()
    return figure

def compare_ion(ion_gofnts, ion_names, ion_flux, ion_err,
                dems, samples, weight, temp, title_name, dem_method='cheby',
                knot_locs=None, sample_num=1000):
    _, med, _ = dems
    ion_temp_fs = temp[np.argmax(med * ion_gofnts, axis=1)]
    ion_model_array = np.zeros((sample_num, len(ion_flux)))
    rand_indices = np.random.choice(
        range(np.shape(samples)[0]), size=sample_num)
    x_arr = np.linspace(-1, 1, len(temp))
    for i, rand_index in enumerate(rand_indices):
        sample = samples[rand_index]
        if dem_method == 'cheby':
            coeffs = sample[:-1]
            s_factor = 10.**sample[-1]
            psi = 10.0**np.polynomial.chebyshev.chebval(x_arr, coeffs)
            ion_model = do_gofnt_matrix_integral(psi, ion_gofnts, temp, weight)
            ion_model_array[i, :] = np.random.normal(
                                                     loc=ion_model,
                                                     scale=(s_factor
                                                            * ion_model))
        elif dem_method == 'gp':
            amp1 = np.exp(sample[-3])
            amp2 = np.exp(sample[-2])
            scale = sample[-1]
            knots = sample[:-3]
            k1 = amp2 * kernels.Matern32(scale) + kernels.Constant(amp1)
            gp = GaussianProcess(k1, jnp.array(knot_locs))
            x_test = np.linspace(-1, 1, len(temp))
            mu, cov = gp.predict(knots, x_test, return_var=True)
            cov = np.array(cov)
            cov[np.where(cov <= 0.0)] = np.min(cov[np.where(cov > 0.0)])
            psi = 10.0**np.random.normal(loc=mu, scale=np.sqrt(cov))
            ion_model_array[i, :] = do_gofnt_matrix_integral(psi, ion_gofnts, temp, weight)

    ion_model_med = np.nanmedian(ion_model_array, axis=0)
    ion_model_low = np.percentile(ion_model_array, 16, axis=0)
    ion_model_upp = np.percentile(ion_model_array, 84, axis=0)
    ion_asym = [(ion_model_med - ion_model_low),
                (ion_model_upp - ion_model_med)]
    plt.errorbar(ion_temp_fs, ion_flux, yerr=ion_err, color='k', marker='.', ls='',
                 label='Data')
    plt.errorbar(ion_temp_fs, ion_model_med, yerr=ion_asym, fmt='None',
                 label='Model')
    for i in range(len(ion_names)):
        ion_name = ion_names[i].split('_')
        new_name = ion_name[0].capitalize() + ' '
        new_name += roman.toRoman(int(ion_name[1]))
        plt.text(ion_temp_fs[i], ion_model_med[i], new_name,
                 bbox=dict(facecolor='gray', alpha=0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(title_name)
    print((ion_model_med - ion_flux) / ion_flux)
    return plt.gcf()


def compare_spec(spec_gofnt, spec_wave, spec_bins, spec_flux, spec_err,
                 samples, weight, temp, title_name, dem_method='cheby',
                 knot_locs=None, sample_num=1000):
    spec_model_array = np.zeros((sample_num, len(spec_flux)))
    rand_indices = np.random.choice(
        range(np.shape(samples)[0]), size=sample_num)
    x_arr = np.linspace(-1, 1, len(temp))
    for i, rand_index in enumerate(rand_indices):
        sample = samples[rand_index]
        if dem_method == 'cheby':
            coeffs = sample[:-1]
            s_factor = 10.**sample[-1]
            psi = 10.0**np.polynomial.chebyshev.chebval(x_arr, coeffs)
            spec_model = do_gofnt_matrix_integral(psi, spec_gofnt, temp, weight)
            spec_model /= spec_bins
            spec_model_array[i, :] = np.random.normal(
                                                    loc=spec_model,
                                                    scale=(s_factor
                                                            * spec_model))
        elif dem_method == 'gp':
            amp1 = np.exp(sample[-3])
            amp2 = np.exp(sample[-2])
            scale = sample[-1]
            knots = sample[:-3]
            k1 = amp1 * kernels.Matern32(scale) + kernels.Constant(amp2)
            gp = GaussianProcess(k1, jnp.array(knot_locs))
            x_test = np.linspace(-1, 1, len(temp))
            mu, cov = gp.predict(knots, x_test, return_var=True)
            cov = np.array(cov)
            cov[np.where(cov <= 0.0)] = np.min(cov[np.where(cov > 0.0)])
            psi = 10.0**np.random.normal(loc=mu, scale=np.sqrt(cov))
            spec_model = do_gofnt_matrix_integral(psi, spec_gofnt, temp, weight)
            spec_model_array[i, :] = spec_model / spec_bins
    spec_model_med = np.median(spec_model_array, axis=0)
    spec_model_low = np.percentile(spec_model_array, 16, axis=0)
    spec_model_upp = np.percentile(spec_model_array, 84, axis=0)
    spec_asym = [(spec_model_med - spec_model_low),
                (spec_model_upp - spec_model_med)]
    plt.errorbar(spec_wave, spec_flux, yerr=spec_err, color='k', marker='.', ls='',
                 label='Data')
    plt.errorbar(spec_wave, spec_model_med, yerr=spec_asym, drawstyle='steps-mid',
                 label='Model')
    plt.legend()
    plt.title(title_name)
    return plt.gcf()