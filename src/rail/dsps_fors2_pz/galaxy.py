#!/bin/env python3

from collections import namedtuple

import jax.numpy as jnp
from jax import jit, vmap

from rail.dsps_fors2_pz import nz_prior_core, prior_alpt0, prior_ft, prior_kt, prior_ktf, prior_pcal, prior_zot

Observation = namedtuple("Observation", ["num", "ref_i_AB", "AB_colors", "AB_colerrs", "valid_filters", "valid_colors", "z_spec"])


def load_galaxy(photometry, ismag, id_i_band=3):
    """load_galaxy _summary_

    :param photometry: fluxes or magnitudes and corresponding errors as read from an ASCII input file
    :type photometry: list or array-like
    :param ismag: whether photometry is provided as AB-magnitudes or fluxes
    :type ismag: bool
    :param id_i_band: index of i-band in the photometry. The default is 3 for LSST u, g, r, i, z, y.
    :type id_i_band: int, optional
    :return: Tuple containing the i-band AB magnitude, the array of color indices for the observations (in AB mag units), the corresponding errors
    and the array of booleans indicating which filters were used for this observation.
    :rtype: tuple
    """
    assert len(photometry) % 2 == 0, "Missing data in observations : check that magnitudes/fluxes and errors are available\n and listed as M (or F), error, M (or F), error, etc."
    _phot = jnp.array([photometry[2 * i] for i in range(len(photometry) // 2)])
    _phot_errs = jnp.array([photometry[2 * i + 1] for i in range(len(photometry) // 2)])

    if ismag:
        c_ab = _phot[:-1] - _phot[1:]
        c_ab_err = jnp.power(jnp.power(_phot_errs[:-1], 2) + jnp.power(_phot_errs[1:], 2), 0.5)
        i_ab = _phot[id_i_band]
        filters_to_use = _phot > 0.0
    else:
        c_ab = -2.5 * jnp.log10(_phot[:-1] / _phot[1:])
        c_ab_err = 2.5 / jnp.log(10) * jnp.power(jnp.power(_phot_errs[:-1] / _phot[:-1], 2) + jnp.power(_phot_errs[1:] / _phot[1:], 2), 0.5)
        i_ab = -2.5 * jnp.log10(_phot[id_i_band]) - 48.6
        filters_to_use = jnp.isfinite(_phot)
    colors_to_use = jnp.array([b1 and b2 for (b1, b2) in zip(filters_to_use[:-1], filters_to_use[1:], strict=True)])
    return i_ab, c_ab, c_ab_err, filters_to_use, colors_to_use


@jit
def col_to_fluxRatio(obs, ref, err):
    r"""col_to_fluxRatio Computes the equivalent data in flux (linear) space from the input in magnitude (logarithmic) space.
    Useful to switch from a $\chi^2$ minimisation in color-space or in flux space.

    :param obs: Observed color index
    :type obs: float or array
    :param ref: Reference (template) color index
    :type ref: float or array
    :param err: Observed noise (aka errors, dispersion)
    :type err: float or array
    :return: $\left( 10^{-0.4 obs}, 10^{-0.4 ref}, 10^{-0.4 err} \right)$
    :rtype: 3-tuple of (float or array)
    """
    obs_f = jnp.power(10.0, -0.4 * obs)
    ref_f = jnp.power(10.0, -0.4 * ref)
    err_f = obs_f * (jnp.power(10.0, -0.4 * err) - 1)  # coindetable
    return obs_f, ref_f, err_f


@jit
def chi_term(obs, ref, err):
    r"""chi_term Compute one term in the $\chi^2$ formula, *i.e.* for one photometric band.

    :param obs: Observed color index
    :type obs: float or array
    :param ref: Reference (template) color index
    :type ref: float or array
    :param err: Observed noise (aka errors, dispersion)
    :type err: float or array
    :return: $\left( \frac{obs-ref}{err} \right)^2$
    :rtype: float or array
    """
    return jnp.power((obs - ref) / err, 2.0)


vmap_chi_term = vmap(chi_term, in_axes=(None, 0, None))  # vmap version to compute the chi value for all colors of a single template, i.e. for all redshifts values


@jit
def z_prior_val(i_mag, zp, nuvk):
    """z_prior_val Computes the prior value for the given combination of observation, template and redshift.

    :param i_mag: Observed magnitude in reference (i) band
    :type i_mag: float
    :param zp: Redshift at which the probability, here the prior, is evaluated
    :type zp: float
    :param nuvk: Templates' NUV-NIR color index, in restframe
    :type nuvk: float
    :return: Prior probability of the redshift zp for this observation, if represented by the given template
    :rtype: float
    """
    alpt0, zot, kt, pcal, ktf_m, ft_m = prior_alpt0(nuvk), prior_zot(nuvk), prior_kt(nuvk), prior_pcal(nuvk), prior_ktf(nuvk), prior_ft(nuvk)
    val_prior = nz_prior_core(zp, i_mag, alpt0, zot, kt, pcal, ktf_m, ft_m)
    return val_prior


vmap_nz_prior = vmap(z_prior_val, in_axes=(None, 0, 0))  # vmap version to compute the prior value for a certain observation and a certain SED template at all redshifts


@jit
def val_neg_log_posterior(z_val, templ_cols, gal_cols, gel_colerrs, gal_iab, templ_nuvk):
    r"""val_neg_log_posterior Computes the negative log posterior (posterior = likelihood * prior) probability of the redshift for an observation, given a template galaxy.
    This corresponds to a reduced $\chi^2$ value in which the prior has been injected.

    :param z_val: Redshift at which the probability, here the posterior, is evaluated
    :type z_val: float
    :param templ_cols: Color indices of the galaxy template
    :type templ_cols: array of floats
    :param gal_cols: Color indices of the observed object
    :type gal_cols: array of floats
    :param gel_colerrs: Color errors/dispersion/noise of the observed object
    :type gel_colerrs: array of floats
    :param gal_iab: Observed magnitude in reference (i) band
    :type gal_iab: float
    :param templ_nuvk: Templates' NUV-NIR color index, in restframe
    :type templ_nuvk: float
    :return: Posterior probability of the redshift zp for this observation, if represented by the given template
    :rtype: float
    """
    _chi = chi_term(gal_cols, templ_cols, gel_colerrs)
    _prior = z_prior_val(gal_iab, z_val, templ_nuvk)
    return jnp.sum(_chi) / len(_chi) - 2 * jnp.log(_prior)


vmap_neg_log_posterior = vmap(val_neg_log_posterior, in_axes=(0, 0, None, None, None, 0))


# @jit
def neg_log_posterior(sps_temp, obs_gal):
    r"""neg_log_posterior Computes the posterior distribution of redshifts (negative log posterior, similar to a $\chi^2$) for a combination of template x observation.

    :param sps_temp: SPS template to be used as reference
    :type sps_temp: SPS_template object (namedtuple)
    :param obs_gal: Observed galaxy
    :type obs_gal: Observation object (namedtuple)
    :return: negative log posterior values along the redshift grid
    :rtype: jax array
    """
    _sel = obs_gal.valid_colors
    neglog_post = vmap_neg_log_posterior(sps_temp.z_grid, sps_temp.colors[:, _sel], obs_gal.AB_colors[_sel], obs_gal.AB_colerrs[_sel], obs_gal.ref_i_AB, sps_temp.nuvk)
    return neglog_post


@jit
def val_neg_log_likelihood(templ_cols, gal_cols, gel_colerrs):
    r"""val_neg_log_likelihood Computes the negative log likelihood of the redshift for an observation, given a template galaxy.
    This is a reduced $\chi^2$ and does not use a prior probability distribution.

    :param templ_cols: Color indices of the galaxy template
    :type templ_cols: array of floats
    :param gal_cols: Color indices of the observed object
    :type gal_cols: array of floats
    :param gel_colerrs: Color errors/dispersion/noise of the observed object
    :type gel_colerrs: array of floats
    :return: Likelihood of the redshift zp for this observation, if represented by the given template.
    :rtype: float
    """
    _chi = chi_term(gal_cols, templ_cols, gel_colerrs)
    return jnp.sum(_chi) / len(_chi)


vmap_neg_log_likelihood = vmap(val_neg_log_likelihood, in_axes=(0, None, None))  # Same as above but for all templates.


# @jit
def neg_log_likelihood(sps_temp, obs_gal):
    r"""neg_log_likelihood Computes the negative log likelihood of redshifts (aka $\chi^2$) for a combination of template x observation.

    :param sps_temp: SPS template to be used as reference
    :type sps_temp: SPS_template object (namedtuple)
    :param obs_gal: Observed galaxy
    :type obs_gal: Observation object (namedtuple)
    :return: negative log likelihood values (aka $\chi^2$) along the redshift grid
    :rtype: jax array
    """
    _sel = obs_gal.valid_colors
    neglog_lik = vmap_neg_log_likelihood(sps_temp.colors[:, _sel], obs_gal.AB_colors[_sel], obs_gal.AB_colerrs[_sel])
    return neglog_lik


def likelihood(sps_temp, obs_gal):
    r"""likelihood Computes the likelihood of redshifts for a combination of template x observation.

    :param sps_temp: SPS template to be used as reference
    :type sps_temp: SPS_template object (namedtuple)
    :param obs_gal: Observed galaxy
    :type obs_gal: Observation object (namedtuple)
    :return: likelihood values (*i.e.* $\exp \left( - \frac{\chi^2}{2} \right)$) along the redshift grid
    :rtype: jax array
    """
    _sel = obs_gal.valid_colors
    neglog_lik = vmap_neg_log_likelihood(sps_temp.colors[:, _sel], obs_gal.AB_colors[_sel], obs_gal.AB_colerrs[_sel])
    return jnp.exp(-0.5 * neglog_lik)


def likelihood_fluxRatio(sps_temp, obs_gal):
    r"""likelihood Computes the likelihood of redshifts for a combination of template x observation.
    Uses the $\chi^2$ distribution in flux-ratio space instead of color space.

    :param sps_temp: SPS template to be used as reference
    :type sps_temp: SPS_template object (namedtuple)
    :param obs_gal: Observed galaxy
    :type obs_gal: Observation object (namedtuple)
    :return: likelihood values (*i.e.* $\exp \left( - \frac{\chi^2}{2} \right)$) along the redshift grid
    :rtype: jax array
    """
    _sel = obs_gal.valid_colors
    obs, ref, err = col_to_fluxRatio(obs_gal.AB_colors[_sel], sps_temp.colors[:, _sel], obs_gal.AB_colerrs[_sel])
    neglog_lik = vmap_neg_log_likelihood(ref, obs, err)
    return jnp.exp(-0.5 * neglog_lik)


# @jit
def posterior(sps_temp, obs_gal):
    r"""posterior Computes the posterior distribution of redshifts for a combination of template x observation.

    :param sps_temp: SPS template to be used as reference
    :type sps_temp: SPS_template object (namedtuple)
    :param obs_gal: Observed galaxy
    :type obs_gal: Observation object (namedtuple)
    :return: posterior probability values (*i.e.* $\exp \left( - \frac{\chi^2}{2} \right) \times prior$) along the redshift grid
    :rtype: jax array
    """
    _sel = obs_gal.valid_colors
    chi2_arr = vmap_neg_log_likelihood(sps_temp.colors[:, _sel], obs_gal.AB_colors[_sel], obs_gal.AB_colerrs[_sel])
    # neglog_lik = chi2_arr - 500.
    _n1 = 10.0 / jnp.max(chi2_arr)
    neglog_lik = _n1 * chi2_arr
    prior_val = vmap_nz_prior(obs_gal.ref_i_AB, sps_temp.z_grid, sps_temp.nuvk)
    # res = jnp.exp(-0.5 * neglog_lik) * prior_val
    res = jnp.power(jnp.exp(-0.5 * neglog_lik), 1 / _n1) * prior_val
    return res


def posterior_fluxRatio(sps_temp, obs_gal):
    r"""posterior Computes the posterior distribution of redshifts for a combination of template x observation.
    Uses the $\chi^2$ distribution in flux-ratio space instead of color space.

    :param sps_temp: SPS template to be used as reference
    :type sps_temp: SPS_template object (namedtuple)
    :param obs_gal: Observed galaxy
    :type obs_gal: Observation object (namedtuple)
    :return: posterior probability values (*i.e.* $\exp \left( - \frac{\chi^2}{2} \right) \times prior$) along the redshift grid
    :rtype: jax array
    """
    _sel = obs_gal.valid_colors
    obs, ref, err = col_to_fluxRatio(obs_gal.AB_colors[_sel], sps_temp.colors[:, _sel], obs_gal.AB_colerrs[_sel])
    chi2_arr = vmap_neg_log_likelihood(ref, obs, err)
    # neglog_lik = chi2_arr - 500.
    _n1 = 10.0 / jnp.max(chi2_arr)
    neglog_lik = _n1 * chi2_arr
    prior_val = vmap_nz_prior(obs_gal.ref_i_AB, sps_temp.z_grid, sps_temp.nuvk)
    # res = jnp.exp(-0.5 * neglog_lik) * prior_val
    res = jnp.power(jnp.exp(-0.5 * neglog_lik), 1 / _n1) * prior_val
    return res
