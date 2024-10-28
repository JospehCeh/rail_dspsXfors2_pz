#!/usr/bin/env python3
#
#  Analysis.py
#
#  Copyright 2023  <joseph@wl-chevalier>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import jax
from jax import numpy as jnp


"""
Reminder :
Cosmo = namedtuple('Cosmo', ['h0', 'om0', 'l0', 'omt'])
sedpyFilter = namedtuple('sedpyFilter', ['name', 'wavelengths', 'transmission'])
BaseTemplate = namedtuple('BaseTemplate', ['name', 'flux', 'z_sps'])
SPS_Templates = namedtuple('SPS_Templates', ['name', 'redshift', 'z_grid', 'i_mag', 'colors', 'nuvk'])
Observation = namedtuple('Observation', ['num', 'AB_fluxes', 'AB_f_errors', 'z_spec'])
DustLaw = namedtuple('DustLaw', ['name', 'EBV', 'transmission'])
"""

# conf_json = 'EmuLP/COSMOS2020-with-FORS2-HSC_only-jax-CC-togglePriorTrue-opa.json' # attention à la localisation du fichier !


@jax.jit
def _cdf(z, pdz):
    cdf = jnp.array([jnp.trapezoid(pdz[:i], x=z[:i]) for i in range(len(z))])
    return cdf


@jax.jit
def _median(z, pdz):
    cdz = _cdf(z, pdz)
    medz = z[jnp.nonzero(cdz >= 0.5, size=1)][0]
    return medz


def extract_pdz(pdf_res, z_grid):
    """extract_pdz Computes and returns the marginilized Probability Density function of redshifts and associated statistics for a single observation ;
    additional point estimates are also computed for each galaxy template for various analytic needs.

    :param pdf_res: Output of photo-z estimation on a single observation
    :type pdf_res: tuple of (dict, float)
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and associated summarized statistics
    :rtype: dict
    """
    pdf_dict = pdf_res[0]
    zs = pdf_res[1]
    pdf_arr = jnp.array([pdf_templ for _, pdf_templ in pdf_dict.items()])
    # print(f"DEBUG extract_pdz : {exp_arr.shape}")
    _n2 = jnp.trapezoid(jnp.nansum(pdf_arr, axis=0), x=z_grid)
    pdf_arr = pdf_arr / _n2
    pdz_dict = {}
    for key, val in pdf_dict.items():
        joint_pdz = val / _n2
        evidence = jnp.trapezoid(joint_pdz, x=z_grid)
        z_ml = z_grid[jnp.nanargmax(joint_pdz)]
        z_avg = jnp.trapezoid(z_grid * joint_pdz / evidence, x=z_grid)
        pdz_dict.update({key: {"evidence_SED": evidence, "z_ML_SED": z_ml, "z_mean_SED": z_avg}})
    pdz = jnp.nansum(pdf_arr, axis=0)
    z_med = _median(z_grid, pdz)
    z_ML = z_grid[jnp.nanargmax(pdz)]
    z_MEAN = jnp.trapezoid(z_grid * pdz, x=z_grid)
    pdz_dict.update({"PDZ": jnp.column_stack((z_grid, pdz)), "z_spec": zs, "z_ML": z_ML, "z_mean": z_MEAN, "z_med": z_med})
    return pdz_dict


def extract_pdz_fromchi2(chi2_res, z_grid):
    r"""extract_pdz_fromchi2 Similar to extract_pdz except takes $\chi^2$ values as inputs - Computes and returns the marginilized Probability Density function of redshifts

    :param chi2_res: Output of photo-z estimation on a single observation, given as the results of $\chi^2$ values, not likelihood.
    :type chi2_res: tuple of (dict, float)
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and *evidence* value for each galaxy template
    :rtype: dict
    """
    chi2_dict = chi2_res[0]
    zs = chi2_res[1]
    chi2_arr = jnp.array([chi2_templ for _, chi2_templ in chi2_dict.items()])
    _n1 = 100.0  # jnp.max(chi2_arr)
    chi2_arr = chi2_arr - _n1  # 10 * chi2_arr / _n1
    exp_arr = jnp.exp(-0.5 * chi2_arr)
    # print(f"DEBUG extract_pdz : {exp_arr.shape}")
    _n2 = jnp.trapezoid(jnp.nansum(exp_arr, axis=0), x=z_grid)
    exp_arr = exp_arr / _n2
    pdz_dict = {}
    for key, val in chi2_dict.items():
        chiarr = val - _n1
        joint_pdz = jnp.exp(-0.5 * chiarr) / _n2
        evidence = jnp.trapezoid(joint_pdz, x=z_grid)
        pdz_dict.update({key: {"SED evidence": evidence}})
    pdz_dict.update({"PDZ": jnp.column_stack((z_grid, jnp.nansum(exp_arr, axis=0))), "z_spec": zs})
    return pdz_dict


def extract_pdz_allseds(pdf_res, z_grid):
    """extract_pdz_allseds Computes and returns the marginilized Probability Density function of redshifts for a single observation ;
    The conditional probability density is also computed for each galaxy template.

    :param pdf_res: Output of photo-z estimation on a single observation
    :type pdf_res: tuple of (dict, float)
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and conditional PDF for each template.
    :rtype: dict
    """
    pdf_dict = pdf_res[0]
    zs = pdf_res[1]
    pdf_arr = jnp.array([pdf_templ for _, pdf_templ in pdf_dict.items()])
    # print(f"DEBUG extract_pdz : {exp_arr.shape}")
    _n2 = jnp.trapezoid(jnp.nansum(pdf_arr, axis=0), x=z_grid)
    pdf_arr = pdf_arr / _n2
    pdz_dict = {}
    for key, val in pdf_dict.items():
        joint_pdz = val / _n2
        evidence = jnp.trapezoid(joint_pdz, x=z_grid)
        pdz_dict.update({key: {"p(z, sed)": joint_pdz, "SED evidence": evidence}})
    pdz_dict.update({"PDZ": jnp.nansum(pdf_arr, axis=0), "z_spec": zs})
    return pdz_dict


def run_from_inputs(inputs):
    """run_from_inputs Run the photometric redshifts estimation with the given input settings.

    :param inputs: Input settings for the photoZ run. Can be loaded from a `JSON` file using `process_fors2.fetchData.json_to_inputs`.
    :type inputs: dict
    :return: Photo-z estimation results. These are not written to disk within this function.
    :rtype: list (tree-like)
    """
    from rail.dsps_fors2_pz import Observation, SPS_Templates, likelihood, likelihood_fluxRatio, load_data_for_run, posterior, posterior_fluxRatio

    z_grid, templates_dict, obs_arr = load_data_for_run(inputs)

    print("Photometric redshift estimation (please be patient, this may take a couple of hours on large datasets) :")

    def has_sps_template(cont):
        return isinstance(cont, SPS_Templates)

    # @partial(jit, static_argnums=1)
    # def estim_zp(observ, prior=True):
    # @jit
    def estim_zp(observ):
        # c = observ.AB_colors[observ.valid_colors]
        # c_err = observ.AB_colerrs[observ.valid_colors]
        if inputs["photoZ"]["prior"] and observ.valid_filters[inputs["photoZ"]["i_band_num"]]:
            probz_dict = (
                jax.tree_util.tree_map(lambda sps_templ: posterior(sps_templ, observ), templates_dict, is_leaf=has_sps_template)
                if inputs["photoZ"]["use_colors"]
                else jax.tree_util.tree_map(lambda sps_templ: posterior_fluxRatio(sps_templ, observ), templates_dict, is_leaf=has_sps_template)
            )
        else:
            probz_dict = (
                jax.tree_util.tree_map(lambda sps_templ: likelihood(sps_templ, observ), templates_dict, is_leaf=has_sps_template)
                if inputs["photoZ"]["use_colors"]
                else jax.tree_util.tree_map(lambda sps_templ: likelihood_fluxRatio(sps_templ, observ), templates_dict, is_leaf=has_sps_template)
            )
        # z_phot_loc = jnp.nanargmin(chi2_arr)
        return probz_dict, observ.z_spec  # chi2_arr, z_phot_loc

    def is_obs(elt):
        return isinstance(elt, Observation)

    tree_of_results_dict = jax.tree_util.tree_map(lambda elt: extract_pdz(estim_zp(elt), z_grid), obs_arr, is_leaf=is_obs)
    print('All done !')

    return tree_of_results_dict
