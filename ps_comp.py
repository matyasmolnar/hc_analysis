"""Power spectrum computations"""


import os

import numpy as np

from pyuvdata import UVData
from pyuvdata import utils as uvutils

import hera_pspec as hp


def pspec_calc(dfile):
    """Power spectrum calculation from visibility file"""

    hera_pkgs = '/users/heramgr/hera_software/' # at NRAO
    if not os.path.exists(hera_pkgs):
        hera_pkgs = '/Users/matyasmolnar/Downloads/HERA_Data/hera_packages/' # local

    # Load beam model
    beamfile = os.path.join(hera_pkgs, 'hera_pspec/hera_pspec/data/HERA_NF_dipole_power.beamfits')
    cosmo = hp.conversions.Cosmo_Conversions()
    uvb = hp.pspecbeam.PSpecBeamUV(beamfile, cosmo=cosmo)

    # Load data into UVData objects
    uvd = UVData()
    uvd.read_uvh5(dfile)

    # find conversion factor from Jy to mK
    Jy_to_mK = uvb.Jy_to_mK(np.unique(uvd.freq_array), pol='xx')

    # reshape to appropriately match a UVData.data_array object and multiply in!
    uvd.data_array *= Jy_to_mK[None, None, :, None]

    # We only have 1 data file here, so slide the time axis by one integration
    # to avoid noise bias (not normally needed!)
    uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)

    # Create a new PSpecData object
    ds = hp.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], beam=uvb)

    # Because we are forming power spectra between datasets that are offset in LST
    # there will be some level of decoherence (and therefore signal loss) of the
    # EoR signal. For short baselines and small LST offsets this is typically
    # negligible, but it is still good to try to recover what coherency we can,
    # simply by phasing (i.e. fringe-stopping) the datasets before forming the power
    # spectra. This can be done with the rephase_to_dset method, and can only be done once.
    ds.rephase_to_dset(0) # Phase to the zeroth dataset

    # change units of UVData objects
    ds.dsets[0].vis_units = 'mK'
    ds.dsets[1].vis_units = 'mK'

    # Find list of baselines pairs to calculate power spectra for
    uvd_ant_copy = uvd.copy()
    uvd_ant_copy.select(times=uvd_ant_copy.time_array[0])

    # Returned values: list of redundant groups, corresponding mean baseline
    # vectors, baseline lengths.
    # No conjugates included, so conjugates is None.
    tol = 0.5  # Tolerance in meters
    baseline_groups, vec_bin_centers, lengths = uvutils.get_baseline_redundancies(\
        uvd_ant_copy.baseline_array, uvd_ant_copy.uvw_array, tol=tol)

    # Selecting shortest (~14.6m) EW baselines group
    # Check to see if baselines haven't already been aggregated by group
    # -> this is done in omnical where only 'only one baseline per unique
    # separation' is kept
    if len(baseline_groups) == len([bl for bl_group in baseline_groups for bl in bl_group]):
        bls_to_include = baseline_groups[0]
        bls1 = [uvutils.baseline_to_antnums(bls_to_include[0], len(uvd.get_ants()))]
        bls2 = bls1
    else:
        bls_to_include = baseline_groups[1]

        # Converting to antnum tuples to be used to construct_blpairs later on
        ant_pairs_to_include = [uvutils.baseline_to_antnums(i, len(uvd.get_ants())) \
                                for i in bls_to_include]
        bls1, bls2, blp = hp.utils.construct_blpairs(ant_pairs_to_include, \
            exclude_permutations=False, exclude_auto_bls=True)

    # Power spectrum calculation
    uvp = ds.pspec(bls1, bls2, (0, 1), [('xx', 'xx')], spw_ranges=[(300, 400), (600,721)], \
        input_data_weight='identity', norm='I', taper='blackman-harris', verbose=False)

    blpairs = np.unique(uvp.blpair_array)
    blps = list(blpairs)

    return uvp, blps
