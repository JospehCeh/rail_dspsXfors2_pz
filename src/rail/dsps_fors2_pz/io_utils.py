#!/usr/bin/env python3
"""
Module to load data for combined SPS and PhotoZ studies within RAIL, applied to LSST.

Created on Wed Oct 24 14:52 2024

@author: joseph
"""

import os
import h5py
import json
import jax
import numpy as np
from tqdm import tqdm
from rail.dsps import load_ssp_templates
from jax import numpy as jnp

_script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    PZDATALOC = os.environ["PZDATALOC"]
except KeyError:
    try:
        PZDATALOC = input("Please type in the path to FORS2 data, e.g. /home/usr/rail_dspsXfors2_pz/src/data")
        os.environ["PZDATALOC"] = PZDATALOC
    except Exception:
        PZDATALOC = os.path.join(_script_dir, "data")
        os.environ["PZDATALOC"] = PZDATALOC

DEFAULTS_DICT = {}
FILENAME_SSP_DATA = "ssp_data_fsps_v3.2_lgmet_age.h5"
# FILENAME_SSP_DATA = "test_fspsData_v3_2_BASEL.h5"
# FILENAME_SSP_DATA = 'test_fspsData_v3_2_C3K.h5'
FULLFILENAME_SSP_DATA = os.path.abspath(os.path.join(PZDATALOC, "ssp", FILENAME_SSP_DATA))
DEFAULTS_DICT.update({"DSPS HDF5": FULLFILENAME_SSP_DATA})

def json_to_inputs(conf_json):
    """
    Load JSON configuration file and return inputs dictionary.

    Parameters
    ----------
    conf_json : path or str
        Path to the configuration file in JSON format.

    Returns
    -------
    dict
        Dictionary of inputs `{param_name: value}`.
    """
    conf_json = os.path.abspath(conf_json)
    with open(conf_json, "r") as inpfile:
        inputs = json.load(inpfile)
    return inputs

def load_ssp(ssp_file=None):
    """load_ssp _summary_

    :param ssp_file: _description_, defaults to None
    :type ssp_file: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    if ssp_file == "" or ssp_file is None or "default" in ssp_file.lower():
        fullfilename_ssp_data = DEFAULTS_DICT["DSPS HDF5"]
    else:
        fullfilename_ssp_data = os.path.abspath(ssp_file)
    ssp_data = load_ssp_templates(fn=fullfilename_ssp_data)
    return ssp_data


def templatesToHDF5(outfilename, templ_dict):
    """
    Writes the SED templates used for photo-z in an HDF5 for a quicker use in future runs.
    Mimics the structure of the class SPS_Templates = namedtuple("SPS_Templates", ["name", "redshift", "z_grid", "i_mag", "colors", "nuvk"]) from process_fors2.photoZ.

    Parameters
    ----------
    outfilename : str or path
        Name of the `HDF5` file that will be written.
    templ_dict : dict
        Dictionary object containing the SED templates.

    Returns
    -------
    path
        Absolute path to the written file - if successful.
    """
    fileout = os.path.abspath(outfilename)

    with h5py.File(fileout, "w") as h5out:
        for key, templ in templ_dict.items():
            groupout = h5out.create_group(key)
            groupout.attrs["name"] = templ.name
            groupout.attrs["redshift"] = templ.redshift
            groupout.create_dataset("z_grid", data=templ.z_grid, compression="gzip", compression_opts=9)
            groupout.create_dataset("i_mag", data=templ.i_mag, compression="gzip", compression_opts=9)
            groupout.create_dataset("colors", data=templ.colors, compression="gzip", compression_opts=9)
            groupout.create_dataset("nuvk", data=templ.nuvk, compression="gzip", compression_opts=9)

    ret = fileout if os.path.isfile(fileout) else f"Unable to write data to {outfilename}"
    return ret


def readTemplatesHDF5(h5file):
    """readTemplatesHDF5 loads the SED templates for photo-z from the specified HDF5 and returns them as a dictionary of objects
    SPS_Templates = namedtuple("SPS_Templates", ["name", "redshift", "z_grid", "i_mag", "colors", "nuvk"]) from process_fors2.photoZ

    :param h5file: Path to the HDF5 containing the SED templates data.
    :type h5file: str or path-like object
    :return: The dictionary of SPS_Templates objects.
    :rtype: dictionary
    """
    from .template import SPS_Templates

    filein = os.path.abspath(h5file)
    out_dict = {}
    with h5py.File(filein, "r") as h5in:
        for key in h5in:
            grp = h5in.get(key)
            out_dict.update(
                {
                    key: SPS_Templates(
                        grp.attrs.get("name"), grp.attrs.get("redshift"), jnp.array(grp.get("z_grid")), jnp.array(grp.get("i_mag")), jnp.array(grp.get("colors")), jnp.array(grp.get("nuvk"))
                    )
                }
            )
    return out_dict


def photoZtoHDF5(outfilename, pz_list):
    """photoZtoHDF5 Saves the pytree of photo-z results (list of dicts) in an HDF5 file.

    :param outfilename: Name of the `HDF5` file that will be written.
    :type outfilename: str or path-like object
    :param pz_list: List of dictionaries containing the photo-z results.
    :type pz_list: list
    :return: Absolute path to the written file - if successful.
    :rtype: str or path-like object
    """
    fileout = os.path.abspath(outfilename)

    with h5py.File(fileout, "w") as h5out:
        for i, posts_dic in enumerate(pz_list):
            groupout = h5out.create_group(f"{i}")
            groupout.create_dataset("PDZ", data=posts_dic.pop("PDZ"), compression="gzip", compression_opts=9)
            groupout.attrs["z_spec"] = posts_dic.pop("z_spec")
            groupout.attrs["z_ML"] = posts_dic.pop("z_ML")
            groupout.attrs["z_mean"] = posts_dic.pop("z_mean")
            groupout.attrs["z_med"] = posts_dic.pop("z_med")
            for templ, tdic in posts_dic.items():
                grp_sed = groupout.create_group(templ)
                grp_sed.attrs["evidence_SED"] = tdic["evidence_SED"]
                grp_sed.attrs["z_ML_SED"] = tdic["z_ML_SED"]
                grp_sed.attrs["z_mean_SED"] = tdic["z_mean_SED"]

    ret = fileout if os.path.isfile(fileout) else f"Unable to write data to {outfilename}"
    return ret


def readPhotoZHDF5(h5file):
    """readPhotoZHDF5 Reads the photo-z results file and generates the corresponding pytree (list of dictionaries) for analysis.

    :param h5file: Path to the HDF5 containing the photo-z results.
    :type h5file: str or path-like object
    :return: List of photo-z results dicts as computed by process_fors2.photoZ.
    :rtype: list
    """
    filein = os.path.abspath(h5file)
    out_list = []
    with h5py.File(filein, "r") as h5in:
        for key, grp in h5in.items():
            obs_dict = {"PDZ": jnp.array(grp.get("PDZ")), "z_spec": grp.attrs.get("z_spec"), "z_ML": grp.attrs.get("z_ML"), "z_mean": grp.attrs.get("z_mean"), "z_med": grp.attrs.get("z_med")}
            for templ, grp_sed in grp.items():
                if "SPEC" in templ:
                    obs_dict.update({templ: dict(grp_sed.attrs.items())})
            out_list.append(obs_dict)
    return out_list


def readDSPSHDF5(h5file):
    """readDSPSHDF5 Reads the contents of the HDF5 that stores the results of DSPS fitting procedure.
    Useful to generate templates for photo-z estimation in `process_fors2.photoZ`.

    :param h5file: Path to the HDF5 file containing the DSPS fitting results.
    :type h5file: str or path-like
    :return: Dictionary of DSPS parameters written as attributes in the HDF5 file
    :rtype: dict
    """
    filein = os.path.abspath(h5file)
    out_dict = {}
    with h5py.File(filein, "r") as h5in:
        for key, grp in h5in.items():
            out_dict.update({key: dict(grp.attrs.items())})
    return out_dict


def _recursive_dict_to_hdf5(group, attrs):
    for key, item in attrs.items():
        if isinstance(item, dict):
            sub_group = group.create_group(key, track_order=True)
            _recursive_dict_to_hdf5(sub_group, item)
        else:
            group.attrs[key] = item

def has_redshift(dic):
    """
    Utility to detect a leaf in a dictionary (tree) based on the assumption that a leaf is a dictionary that contains individual information linked to a spectrum, such as the redshift of the galaxy.

    Parameters
    ----------
    dic : dictionary
        Dictionary with data. Within the context of this function's use, this is an output of the catering of data to fit on DSPS.
        This function is applied to a global dictionary (tree) and its sub-dictionaries (leaves - as identified by this function).

    Returns
    -------
    bool
        `True` if `'redshift'` is in `dic.keys()` - making it a leaf - `False` otherwise.
    """
    return "redshift" in list(dic.keys())

def load_data_for_run(inp_glob):
    """load_data_for_run Generates input data from the inputs configuration dictionary

    :param inp_glob: input configuration and settings
    :type inp_glob: dict
    :return: data for photo-z evaluation : redshift grid, templates dictionary and the list of observations
    :rtype: tuple of jax.ndarray, dictionary, list
    """
    from rail.dsps_fors2_pz import NIR_filt, NUV_filt, Observation, get_2lists, load_filt, load_galaxy, make_legacy_templates, make_sps_templates, sedpyFilter

    _ssp_file = (
        None
        if (inp_glob["photoZ"]["ssp_file"].lower() == "default" or inp_glob["photoZ"]["ssp_file"] == "" or inp_glob["photoZ"]["ssp_file"] is None)
        else os.path.abspath(inp_glob["photoZ"]["ssp_file"])
    )
    ssp_data = load_ssp(_ssp_file)

    inputs = inp_glob["photoZ"]
    z_grid = jnp.arange(inputs["Z_GRID"]["z_min"], inputs["Z_GRID"]["z_max"] + inputs["Z_GRID"]["z_step"], inputs["Z_GRID"]["z_step"])

    filters_dict = inputs["Filters"]
    for _f in filters_dict:
        filters_dict[_f]["path"] = os.path.abspath(os.path.join(PZDATALOC, filters_dict[_f]["path"]))
    print("Loading filters :")
    filters_arr = tuple(sedpyFilter(*load_filt(int(ident), filters_dict[ident]["path"], filters_dict[ident]["transmission"])) for ident in tqdm(filters_dict)) + (NUV_filt, NIR_filt)
    N_FILT = len(filters_arr) - 2
    # print(f"DEBUG: filters = {filters_arr}")

    print("Building templates :")
    Xfilt = get_2lists(filters_arr)
    # sps_temp_pkl = os.path.abspath(inputs["Templates"])
    # sps_par_dict = read_params(sps_temp_pkl)
    if inputs["Templates"]["overwrite"] or not os.path.isfile(os.path.abspath(inputs["Templates"]["output"])):
        sps_temp_h5 = os.path.abspath(os.path.join(PZDATALOC, inputs["Templates"]["input"]))
        sps_par_dict = readDSPSHDF5(sps_temp_h5)
        if "sps" in inputs["Mode"].lower():
            templ_dict = jax.tree_util.tree_map(lambda dico: make_sps_templates(dico, Xfilt, z_grid, ssp_data, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)
        else:
            templ_dict = jax.tree_util.tree_map(lambda dico: make_legacy_templates(dico, Xfilt, z_grid, ssp_data, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)
        _ = templatesToHDF5(inputs["Templates"]["output"], templ_dict)
    else:
        templ_dict = readTemplatesHDF5(inputs["Templates"]["output"])

    print("Loading observations :")
    data_path = os.path.abspath(os.path.join(PZDATALOC, inputs["Dataset"]["path"]))
    data_ismag = inputs["Dataset"]["type"].lower() == "m"

    data_file_arr = np.loadtxt(data_path)
    obs_arr = []

    for i in tqdm(range(data_file_arr.shape[0])):
        try:
            assert (len(data_file_arr[i, :]) == 1 + 2 * N_FILT) or (
                len(data_file_arr[i, :]) == 1 + 2 * N_FILT + 1
            ), f"At least one filter is missing in datapoint {data_file_arr[i,0]} : length is {len(data_file_arr[i,:])}, {1+2*N_FILT} values expected.\nDatapoint removed from dataset."
            # print(int(data_file_arr[i, 0]))
            if len(data_file_arr[i, :]) == 1 + 2 * N_FILT + 1:
                observ = Observation(int(data_file_arr[i, 0]), *load_galaxy(data_file_arr[i, 1 : 2 * N_FILT + 1], data_ismag, id_i_band=inputs["i_band_num"]), data_file_arr[i, 2 * N_FILT + 1])
            else:
                observ = Observation(int(data_file_arr[i, 0]), *load_galaxy(data_file_arr[i, 1 : 2 * N_FILT + 1], data_ismag, id_i_band=inputs["i_band_num"]), jnp.nan)
            # print(observ.num)
            obs_arr.extend([observ])
        except AssertionError:
            pass
    return z_grid, templ_dict, obs_arr
