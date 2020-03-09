"""
Codes for creating and manipulating gate filters.

@title: filtering
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 20/11/2017
@modification: 25/02/2020

.. autosummary::
    :toctree: generated/

    texture
    do_gatefilter_cpol
    do_gatefilter
"""
# Libraries
import pyart
import netCDF4
import numpy as np


def texture(data):
    """
    Compute the texture of data.
    Compute the texture of the data by comparing values with a 3x3 neighborhood
    (based on :cite:`Gourley2007`). NaN values in the original array have
    NaN textures. (Wradlib function)

    Parameters:
    ==========
    data : :class:`numpy:numpy.ndarray`
        multi-dimensional array with shape (..., number of beams, number
        of range bins)

    Returns:
    =======
    texture : :class:`numpy:numpy.ndarray`
        array of textures with the same shape as data
    """
    x1 = np.roll(data, 1, -2)  # center:2
    x2 = np.roll(data, 1, -1)  # 4
    x3 = np.roll(data, -1, -2)  # 8
    x4 = np.roll(data, -1, -1)  # 6
    x5 = np.roll(x1, 1, -1)  # 1
    x6 = np.roll(x4, 1, -2)  # 3
    x7 = np.roll(x3, -1, -1)  # 9
    x8 = np.roll(x2, -1, -2)  # 7

    # at least one NaN would give a sum of NaN
    xa = np.array([x1, x2, x3, x4, x5, x6, x7, x8])

    # get count of valid neighboring pixels
    xa_valid = np.ones(np.shape(xa))
    xa_valid[np.isnan(xa)] = 0
    # count number of valid neighbors
    xa_valid_count = np.sum(xa_valid, axis=0)

    num = np.zeros(data.shape)
    for xarr in xa:
        diff = data - xarr
        # difference of NaNs will be converted to zero
        # (to not affect the summation)
        diff[np.isnan(diff)] = 0
        # only those with valid values are considered in the summation
        num += diff ** 2

    # reinforce that NaN values should have NaN textures
    num[np.isnan(data)] = np.nan

    return np.sqrt(num / xa_valid_count)


def get_gatefilter_GMM(radar, refl_name='DBZ', vel_name='VEL', phidp_name='PHIDP', zdr_name='ZDR'):
    # Cutoff
    r = radar.range['data']
    azi = radar.azimuth['data']
    [R, A] = np.meshgrid(r, azi)
    cutoff = 10 * np.log10(4e-5 * R)

    refl = radar.fields[refl_name]['data'].copy()
    refl[refl < cutoff] = np.NaN
    radar.add_field_like(refl_name, 'NPOS', refl, replace_existing=True)

    # GMM clustering (indpdt from cutoff)
    cluster = get_clustering(radar)
    radar.add_field_like(refl_name, 'CLUS', cluster, replace_existing=True)

    pos = (cluster == 1) & (radar.fields[refl_name]['data'] < 20)
    radar.add_field_like(refl_name, 'TPOS', pos, replace_existing=True)

    # Using GMM and Cutoff to filter.
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_equal('CLUS', 5)
    gf.exclude_equal('CLUS', 2)
    gf.exclude_equal('TPOS', 1)
    gf.exclude_invalid('NPOS')
    gf = pyart.correct.despeckle_field(radar, 'DBZ', gatefilter=gf)

    return gf


def do_gatefilter_cpol(radar,
                       refl_name='DBZ',
                       phidp_name="PHIDP",
                       rhohv_name='RHOHV_CORR',
                       zdr_name="ZDR",
                       snr_name='SNR',
                       vel_name='VEL'):
    """
    Filtering function adapted to CPOL.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        rhohv_name: str
            Cross correlation ratio field name.
        ncp_name: str
            Name of the normalized_coherent_power field.
        zdr_name: str
            Name of the differential_reflectivity field.

    Returns:
    ========
        gf_despeckeld: GateFilter
            Gate filter (excluding all bad data).
    """
    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'].replace("since", "since "))

    if radar_start_date.year < 2009:
        return get_gatefilter_GMM(radar, 
                                  refl_name=refl_name, 
                                  vel_name=vel_name, 
                                  phidp_name=phidp_name, 
                                  zdr_name=zdr_name)

    r = radar.range['data']
    azi = radar.azimuth['data']
    R, _ = np.meshgrid(r, azi)
    refl = radar.fields[refl_name]['data'].copy()
    fcut = 10 * np.log10(4e-5 * R)
    refl[refl < fcut] = np.NaN
    radar.add_field_like(refl_name, 'NDBZ', refl)

    gf = pyart.filters.GateFilter(radar)
    gf.exclude_invalid('NDBZ')
    gf.exclude_below(snr_name, 9)

    gf.exclude_outside(zdr_name, -3.0, 7.0)
    gf.exclude_outside(refl_name, -20.0, 80.0)

    # dphi = texture(radar.fields[phidp_name]['data'])
    # radar.add_field_like(phidp_name, 'PHITXT', dphi)
    # gf.exclude_above('PHITXT', 20)
    if radar_start_date.year > 2007:
        gf.exclude_below(rhohv_name, 0.5)

    # Remove rings in march 1999.
    if radar_start_date.year == 1999 and radar_start_date.month == 3:
        radar.add_field_like(refl_name, 'RRR', R)
        gf.exclude_above('RRR', 140e3)
        radar.fields.pop('RRR')

    gf_despeckeld = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    # Remove tmp fields.
    try:
        radar.fields.pop('NDBZ')
        # radar.fields.pop('PHITXT')
    except Exception:
        pass

    return gf_despeckeld


def do_gatefilter(radar, refl_name='DBZ', phidp_name="PHIDP", rhohv_name='RHOHV_CORR', zdr_name="ZDR", snr_name='SNR'):
    """
    Basic filtering function for dual-polarisation data.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_name: str
            Reflectivity field name.
        rhohv_name: str
            Cross correlation ratio field name.
        ncp_name: str
            Name of the normalized_coherent_power field.
        zdr_name: str
            Name of the differential_reflectivity field.

    Returns:
    ========
        gf_despeckeld: GateFilter
            Gate filter (excluding all bad data).
    """
    # Initialize gatefilter
    gf = pyart.filters.GateFilter(radar)

    # Remove obviously wrong data.
    gf.exclude_outside(zdr_name, -6.0, 7.0)
    gf.exclude_outside(refl_name, -20.0, 80.0)

    # Compute texture of PHIDP and remove noise.
    dphi = texture(radar.fields[phidp_name]['data'])
    radar.add_field_like(phidp_name, 'PHITXT', dphi)
    gf.exclude_above('PHITXT', 20)
    gf.exclude_below(rhohv_name, 0.6)

    # Despeckle
    gf_despeckeld = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    try:
        # Remove PHIDP texture
        radar.fields.pop('PHITXT')
    except Exception:
        pass

    return gf_despeckeld
