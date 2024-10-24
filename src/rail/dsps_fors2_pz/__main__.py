#!/usr/bin/env python3
#
#  __main__.py
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

import sys
import jax
from rail.dsps_fors2_pz import Observation, SPS_Templates, extract_pdz, json_to_inputs, likelihood, likelihood_fluxRatio, load_data_for_run, posterior, posterior_fluxRatio


def main(args):
    """
    Main function to start an external call to the photoZ module. Arguments must be the JSON configuration file.
    """
    conf_json = args[1] if len(args) > 1 else "./defaults.json"  # le premier argument de args est toujours `__main__.py` ; attention à la localisation du fichier !
    inputs = json_to_inputs(conf_json)

    z_grid, templates_dict, obs_arr = load_data_for_run(inputs)

    """Dust and Opacity are normally included in DSPS calculations
    ebvs_in_use = jnp.array([d.EBV for d in dust_arr])
    laws_in_use = jnp.array([0 if d.name == "Calzetti" else 1 for d in dust_arr])

    _old_dir = os.getcwd()
    _path = os.path.abspath(__file__)
    _dname = os.path.dirname(_path)
    os.chdir(_dname)
    opa_path = os.path.abspath(inputs['Opacity'])
    #ebv_prior_file = inputs['E(B-V) prior file']
    #ebv_prior_df = pd.read_pickle(ebv_prior_file)
    #cols_to_stack = tuple(ebv_prior_df[col].values for col in ebv_prior_df.columns)
    #ebv_prior_arr = jnp.column_stack(cols_to_stack)
    os.chdir(_old_dir)

    _selOpa = (wl_grid < 1300.)
    wls_opa = wl_grid[_selOpa]
    opa_zgrid, opacity_grid = extinction.load_opacity(opa_path, wls_opa)
    extrap_ones = jnp.ones((len(z_grid), len(wl_grid)-len(wls_opa)))
    """

    print("Photometric redshift estimation :")

    def has_sps_template(cont):
        """has_sps_template _summary_

        :param cont: _description_
        :type cont: _type_
        :return: _description_
        :rtype: _type_
        """
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

    if inputs["photoZ"]["save results"]:
        from .io_utils import photoZtoHDF5

        # df_gal.to_pickle(f"{inputs['run name']}_results_summary.pkl")
        # with open(f"{inputs['photoZ']['run name']}_posteriors_dict.pkl", "wb") as handle:
        #    pickle.dump(tree_of_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        resfile = photoZtoHDF5(f"{inputs['photoZ']['run name']}_posteriors_dict.h5", tree_of_results_dict)
        print(resfile)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
