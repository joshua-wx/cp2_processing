"""
Codes for correcting Doppler velocity.

@title: velocity
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 11/12/2017
@date: 08/02/2020

.. autosummary::
    :toctree: generated/

    _check_nyquist_velocity
    unravel
"""
import pyart
import numpy as np


def _check_nyquist_velocity(radar, vel_name='VEL'):
    """
    Check if Nyquist velocity is present in the instrument parameters. If not,
    then it is created.
    """

    n_sweeps = len(radar.fixed_angle['data'])
    nyquist_rays = radar.instrument_parameters['nyquist_velocity']['data']
    nquist_sweeps = []
    for i in range(n_sweeps):
        sweep_idx = radar.get_start(i)
        nquist_sweeps.append(nyquist_rays[sweep_idx])

    return nquist_sweeps


def unravel(radar, gatefilter, vel_name='VEL', dbz_name='DBZ'):
    """
    Unfold Doppler velocity using Py-ART region based algorithm. Automatically
    searches for a folding-corrected velocity field.

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    gatefilter:
        Filter excluding non meteorological echoes.
    vel_name: str
        Name of the (original) Doppler velocity field.
    dbz_name: str
        Name of the reflecitivity field.

    Returns:
    ========
    vel_meta: dict
        Unfolded Doppler velocity.
    """
    import unravel

    vnyq = _check_nyquist_velocity(radar, vel_name)

    unfvel = unravel.unravel_3D_pyart(radar,
                                      vel_name,
                                      dbz_name,
                                      gatefilter=gatefilter,
                                      alpha=0.8,
                                      nyquist_velocity=vnyq,
                                      strategy='long_range')

    vel_meta = pyart.config.get_metadata('velocity')
    vel_meta['data'] = np.ma.masked_where(gatefilter.gate_excluded, unfvel).astype(np.float32)
    vel_meta['_Least_significant_digit'] = 2
    vel_meta['_FillValue'] = np.NaN
    vel_meta['comment'] = 'UNRAVEL algorithm.'
    vel_meta['long_name'] = 'Doppler radial velocity of scatterers away from instrument'
    vel_meta['standard_name'] = 'radial_velocity_of_scatterers_away_from_instrument'
    vel_meta['units'] = 'm s-1'

    try:
        vel_meta.pop('standard_name')
    except Exception:
        pass

    return vel_meta
