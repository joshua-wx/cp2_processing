"""
Codes for correcting and estimating various radar and meteorological parameters.

@title: radar_codes
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 04/04/2017
@date: 14/04/2020

.. autosummary::
    :toctree: generated/

    _my_snr_from_reflectivity
    _nearest
    check_azimuth
    check_reflectivity
    check_year
    correct_rhohv
    correct_zdr
    get_radiosoundings
    read_radar
    snr_and_sounding
"""
# Python Standard Library
import os
import re
import glob
import time
import fnmatch
from datetime import datetime

# Other Libraries
import pyart
import scipy
import cftime
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

def read_csv(csv_ffn, header_line):
    """
    CSV reader used for the radar locations file (comma delimited)
    
    Parameters:
    ===========
        csv_ffn: str
            Full filename to csv file
            
        header_line: int or None
            to use first line of csv as header = 0, use None to use column index
            
    Returns:
    ========
        as_dict: dict
            csv columns are dictionary
    
    """
    df = pd.read_csv(csv_ffn, header=header_line)
    as_dict = df.to_dict(orient='list')
    return as_dict


def read_cal_file(cal_ffn, radar_dt):
    """
    read Z or ZDR calibration csv file into dictionary
    
    Parameters:
    ===========
        cal_ffn: str
            Full filename to csv file
            
    Returns:
    ========
        cal_dict: dict
            calibration data as dictionary
    
    """
    #load calibration file
    dict_in = read_csv(cal_ffn, None)
    cal_dict = {}
    #rename dictonary fields
    cal_dict['cal_start'] = np.array([datetime.strptime(str(date), '%Y%m%d').date() for date in dict_in[0]])
    cal_dict['cal_end']   = np.array([datetime.strptime(str(date), '%Y%m%d').date() for date in dict_in[1]])
    cal_dict['cal_mean']  = np.array(list(map(float, dict_in[2])))
    
    #find offset value
    offset = 0
    if cal_dict:
        caldt_mask  = np.logical_and(radar_dt.date()>=cal_dict['cal_start'], radar_dt.date()<=cal_dict['cal_end'])
        offset_match = cal_dict['cal_mean'][caldt_mask]
        if len(offset_match) == 0:
            print('time period not found in cal file')
        elif len(offset_match) > 1:
            print('calibration data error (multiple matches)')                
        else:
            offset = float(offset_match)
    else:
        error_msg = print('no cal file found')
        offset = 0
    
    #return
    return offset


def snr_from_reflectivity(radar, refl_field='DBZ'):
    """
    Just in case pyart.retrieve.calculate_snr_from_reflectivity, I can calculate
    it 'by hand'.
    Parameter:
    ===========
    radar:
        Py-ART radar structure.
    refl_field_name: str
        Name of the reflectivity field.
    Return:
    =======
    snr: dict
        Signal to noise ratio.
    """
    range_grid, _ = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    range_grid += 1  # Cause of 0

    # remove range scale.. This is basically the radar constant scaled dBm
    pseudo_power = (radar.fields[refl_field]['data'] - 20.0 * np.log10(range_grid / 1000.0))
    # The noise_floor_estimate can fail sometimes in pyart, that's the reason
    # why this whole function exists.
    noise_floor_estimate = -40
    snr_data = pseudo_power - noise_floor_estimate

    return snr_data

def _nearest(items, pivot):
    """
    Find the nearest item.

    Parameters:
    ===========
        items:
            List of item.
        pivot:
            Item we're looking for.

    Returns:
    ========
        item:
            Value of the nearest item found.
    """
    return min(items, key=lambda x: abs(x - pivot))


def check_reflectivity(radar, refl_field_name='DBZ'):
    """
    Checking if radar has a proper reflectivity field.  It's a minor problem
    concerning a few days in 2011 for CPOL.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_field_name: str
            Name of the reflectivity field.

    Return:
    =======
    True if radar has a non-empty reflectivity field.
    """
    dbz = radar.fields[refl_field_name]['data']

    if np.ma.isMaskedArray(dbz):
        if dbz.count() == 0:
            # Reflectivity field is empty.
            return False

    return True

def correct_rhohv(radar, rhohv_name='RHOHV', snr_name='SNR'):
    """
    Correct cross correlation ratio (RHOHV) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 5)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        rhohv_name: str
            Cross correlation field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        rho_corr: array
            Corrected cross correlation ratio.
    """
    rhohv = radar.fields[rhohv_name]['data'].copy()
    snr = radar.fields[snr_name]['data'].copy()

    #appears to be done onsite at CP2
#     natural_snr = 10**(0.1 * snr)
#     natural_snr = natural_snr.filled(-9999)
#     rho_corr = rhohv * (1 + 1 / natural_snr)
    rho_corr = rhohv
    
    # Not allowing the corrected RHOHV to be lower than the raw rhohv
    rho_corr[np.isnan(rho_corr) | (rho_corr < 0) | (rho_corr > 1)] = 1
    try:
        rho_corr = rho_corr.filled(1)
    except Exception:
        pass

    return np.ma.masked_array(rho_corr, rhohv.mask)


def correct_standard_name(radar):
    """
    'standard_name' is a protected keyword for metadata in the CF conventions.
    To respect the CF conventions we can only use the standard_name field that
    exists in the CF table.

    Parameter:
    ==========
    radar: Radar object
        Py-ART data structure.
    """
    try:
        radar.range.pop('standard_name')
        radar.azimuth.pop('standard_name')
        radar.elevation.pop('standard_name')
    except Exception:
        pass

    try:
        radar.sweep_number.pop('standard_name')
        radar.fixed_angle.pop('standard_name')
        radar.sweep_mode.pop('standard_name')
    except Exception:
        pass

    good_keys = ['corrected_reflectivity', 'total_power', 'radar_estimated_rain_rate', 'corrected_velocity']
    for k in radar.fields.keys():
        if k not in good_keys:
            try:
                radar.fields[k].pop('standard_name')
            except Exception:
                continue

    try:
        radar.fields['velocity']['standard_name'] = 'radial_velocity_of_scatterers_away_from_instrument'
        radar.fields['velocity']['long_name'] = 'Doppler radial velocity of scatterers away from instrument'
    except KeyError:
        pass

    radar.latitude['standard_name'] = 'latitude'
    radar.longitude['standard_name'] = 'longitude'
    radar.altitude['standard_name'] = 'altitude'

    return None


def correct_zdr(radar, zdr_name='ZDR', snr_name='SNR'):
    """
    Correct differential reflectivity (ZDR) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 6)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        zdr_name: str
            Differential reflectivity field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        corr_zdr: array
            Corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]['data'].copy()
    snr = radar.fields[snr_name]['data'].copy()
    alpha = 1.48
    natural_zdr = 10 ** (0.1 * zdr)
    natural_snr = 10 ** (0.1 * snr)
    corr_zdr = 10 * np.log10((alpha * natural_snr * natural_zdr) / (alpha * natural_snr + alpha - natural_zdr))

    return corr_zdr


def coverage_content_type(radar):
    """
    Adding metadata for compatibility with ACDD-1.3

    Parameter:
    ==========
    radar: Radar object
        Py-ART data structure.
    """
    radar.range['coverage_content_type'] = 'coordinate'
    radar.azimuth['coverage_content_type'] = 'coordinate'
    radar.elevation['coverage_content_type'] = 'coordinate'
    radar.latitude['coverage_content_type'] = 'coordinate'
    radar.longitude['coverage_content_type'] = 'coordinate'
    radar.altitude['coverage_content_type'] = 'coordinate'

    radar.sweep_number['coverage_content_type'] = 'auxiliaryInformation'
    radar.fixed_angle['coverage_content_type'] = 'auxiliaryInformation'
    radar.sweep_mode['coverage_content_type'] = 'auxiliaryInformation'

    for k in radar.fields.keys():
        if k == 'radar_echo_classification':
            radar.fields[k]['coverage_content_type'] = 'thematicClassification'
        elif k in ['normalized_coherent_power', 'normalized_coherent_power_v']:
            radar.fields[k]['coverage_content_type'] = 'qualityInformation'
        else:
            radar.fields[k]['coverage_content_type'] = 'physicalMeasurement'

    return None


def read_radar(radar_file_name):
    """
    Read the input radar file.

    Parameter:
    ==========
        radar_file_name: str
            Radar file name.

    Return:
    =======
        radar: struct
            Py-ART radar structure.
    """
    # Read the input radar file.
    try:
        #load radar
        radar = pyart.io.read_mdv(radar_file_name, file_field_names=True)
        
    except Exception:
        raise
        
    radar.fields['VEL']['units'] = "m s-1"
    return radar


def temperature_profile(radar):
    """
    Compute the signal-to-noise ratio as well as interpolating the radiosounding
    temperature on to the radar grid. The function looks for the radiosoundings
    that happened at the closest time from the radar. There is no time
    difference limit.
    Parameters:
    ===========
    radar:
        Py-ART radar object.
    Returns:
    ========
    z_dict: dict
        Altitude in m, interpolated at each radar gates.
    temp_info_dict: dict
        Temperature in Celsius, interpolated at each radar gates.
    """
    #set era path
    era5_root = '/g/data/ub4/era5/netcdf/pressure'
    
    #extract request data
    request_lat = radar.latitude['data'][0]
    request_lon = radar.longitude['data'][0]
    request_dt = pd.Timestamp(cftime.num2pydate(radar.time['data'][0], radar.time['units']))
    
    #build file paths
    month_str = request_dt.month
    year_str = request_dt.year
    temp_ffn = glob.glob(f'{era5_root}/t/{year_str}/t_era5_aus_{year_str}{month_str:02}*.nc')[0]
    geop_ffn = glob.glob(f'{era5_root}/z/{year_str}/z_era5_aus_{year_str}{month_str:02}*.nc')[0]
    
    #extract data
    with xr.open_dataset(temp_ffn) as temp_ds:
        temp_data = temp_ds.t.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data[:] - 273.15 #units: deg K -> C
    with xr.open_dataset(geop_ffn) as geop_ds:
        geop_data = geop_ds.z.sel(longitude=request_lon, method='nearest').sel(latitude=request_lat, method='nearest').sel(time=request_dt, method='nearest').data[:]/9.80665 #units: m**2 s**-2 -> m
    #flipdata (ground is first row)
    temp_data = np.flipud(temp_data)
    geop_data = np.flipud(geop_data)

    #append surface data using lowest level
    geop_data = np.append(geop_data,[0])
    temp_data = np.append(temp_data, temp_data[-1])
        
    z_dict, temp_dict = pyart.retrieve.map_profile_to_gates(temp_data, geop_data, radar)

    temp_info_dict = {'data': temp_dict['data'],  # Switch to celsius.
                      'long_name': 'Sounding temperature at gate',
                      'standard_name': 'temperature',
                      'valid_min': -100, 'valid_max': 100,
                      'units': 'degrees Celsius',
                      'comment': 'Radiosounding date: %s' % (request_dt.strftime("%Y/%m/%d"))}

    return z_dict, temp_info_dict