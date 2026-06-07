import numpy as np
import os
import astropy.units as u
import pandas as pd
from datetime import datetime, timedelta
from astropy.time import Time
import re
from sunpy.coordinates import sun
import helio_coords as hcoords
import matplotlib.pyplot as plt

import joblib
import onnxruntime as ort

import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA
import huxt_insitu as His

#===============================================================================
# <codecell> Import ICME-CME data

# Function to convert a date string to a datetime object
def convert_to_datetime(date_str):
    # List of possible date formats
    date_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H%M',
        '%Y/%m/%d %H%M(%S)',
        '%Y/%m/%d %H%M(S)',
        '%Y/%m/%d %H:%M',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M(S)',
        '%d/%m/%Y %H:%M'
    ]
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    # If no format matches, return None
    return None
    
# Function to convert Time_Error from string to timedelta
def convert_to_timedelta(time_str):
    try:
        return pd.to_timedelta(time_str)
    except ValueError:
        return None
        
# Function to get earth's latitude
def get_earth_lat(dt):
    """
    A function to return Earth latitude for a given date, in radians
 
    Parameters
    ----------
    dt : datetime
 
    Returns
    -------
    E_lat: Earth latitude, with astropy units of radians
 
    """
    cr, cr_lon_init = Hin.datetime2huxtinputs(dt)
    
    # Use the HUXt ephemeris data to get Earth lat over the CR
    # ========================================================
    dummymodel = H.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=0.1*u.day,
                         cr_num=cr,cr_lon_init=cr_lon_init, lon_out=0.0*u.deg)
    # retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')
    # get average Earth lat
    E_lat = np.nanmean(earth.lat_c)
    E_lat = E_lat.to(u.deg)  # Convert to degrees
    return E_lat
        
# Function to plot time series of solar wind speed at earth with asw and cme
def plot_earth_timeseries(model, modelbg, plot_omni=True):
    """
    A function to plot the HUXt Earth time series. With option to download and plot OMNI data.
    Args:
        model : input model class
        plot_omni: Boolean, if True downloads and plots OMNI data

    Returns:
        fig : Figure handle
        axs : Axes handles

    """

    huxt_ts = HA.get_observer_timeseries(model, observer='Earth')
    ts_bg = HA.get_observer_timeseries(modelbg, observer='Earth')
    cme_out = model.cmes[0]
    stats = cme_out.compute_arrival_at_location(0.0 * u.rad, 215.0 * u.solRad)
    hit = stats['hit']
    if hit == 1:
        t_arrive = stats['t_arrive'].value
        t_arrive = Time(t_arrive, format='jd', scale='utc')

    # 2-panel plot if the B polarity has been traced
    if hasattr(model, 'b_grid'):
        fig, axs = plt.subplots(2, 1, figsize=(14, 7))
        axs[1].plot(huxt_ts['time'], np.sign(huxt_ts['bpol']), 'k.', label='HUXt')
        axs[1].set_ylabel('B polarity')
    else:
        fig, axs = plt.subplots(1, 1, figsize=(14, 4))
        axs = np.array([axs])

    axs[0].plot(huxt_ts['time'], huxt_ts['vsw'], 'k', label='HUXt')
    axs[0].plot(ts_bg['time'], ts_bg['vsw'], 'k', linestyle='dotted', label='No CME')
    if  hit == 1:
        axs[0].axvline(t_arrive.datetime,color='k')
    axs[0].set_ylim(250, 1000)

    starttime = huxt_ts['time'][0]
    endtime = huxt_ts['time'][len(huxt_ts)//2]

    if plot_omni:
        # grab the omni data
        data = Hin.get_omni(starttime, endtime)
        # plot the period of interest
        mask = (data['datetime'] >= starttime) & (data['datetime'] <= endtime)
        plotdata = data[mask]
        axs[0].plot(plotdata['datetime'], plotdata['V'], 'r', label='OMNI')

        if hasattr(model, 'b_grid'):
            axs[1].plot(plotdata['datetime'], -np.sign(plotdata['BX_GSE']) * 0.92, 'r.', label='OMNI')
            axs[1].set_ylim(-1.1, 1.1)

    for a in axs:
        a.set_xlim(starttime, endtime)
        a.legend()

    axs[0].set_ylabel('Solar Wind Speed (km/s)')

    if axs.size == 1:
        axs[0].set_xlabel('Date')
    elif axs.size == 2:
        axs[0].set_xticklabels([])
        axs[1].set_xlabel('Date')

    fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)

    return fig, axs
    
# --- Load data ----------------------------------------------------------------
# Read in Blair's pairing on DONKI and CR2003
project_dirs = H._setup_dirs_()
crpath = os.path.join(project_dirs['input'],'(I)CMEs.csv')

# Load the CSV file into a DataFrame
crlist = pd.read_csv(crpath)

# Convert date columns to datetime objects using datetime library
date_columns = ['Time_21.5', 'Disturbance_Time']
for column in date_columns:
    crlist[column] = crlist[column].apply(lambda x: convert_to_datetime(x) if isinstance(x, str) else None)

# compute 21.5-215 transit time
crlist['tt_21'] = np.nan 
for irow in range(0, len(crlist)):
    crlist.loc[irow,'tt_21'] = crlist.loc[irow,'Disturbance_Time'] - crlist.loc[irow,'Time_21.5']
# Convert  from timedelta to days
crlist['tt_21'] = crlist['tt_21'].apply(lambda x: x.days + x.seconds / 86400 if isinstance(x, timedelta) else None)

# Add a new column with carrington rotation number of CME time
crlist['cr_num'] = crlist['Time_21.5'].apply(lambda dt: Hin.datetime2huxtinputs(dt)[0])
crlist['cr_lon_init'] = crlist['Time_21.5'].apply(lambda dt: Hin.datetime2huxtinputs(dt)[1])
crlist['earth_lat'] = crlist['Time_21.5'].apply(lambda dt: get_earth_lat(dt))

#===============================================================================

#HUXt run parameters
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 230*u.solRad #outer boundary for HUXt runs

# Iterate over all CMEs in crlist
for _, onecme in crlist.iterrows():
    
    #========================================================
    # Obtain OMNI boundary conditions at 21.5 rS
    start_time = onecme['Time_21.5']

    dl_starttime = start_time - timedelta(days=27.27)
    dl_endtime = start_time

    omni = Hin.get_omni(dl_starttime, dl_endtime)

    omni_noicmes = His.removeICMEs(omni,
                    icme_list = 'CaneRichardson',
                    pre_icme_buffer = 0.2, #days
                    post_icme_buffer = 1, #days
                    interp_gaps = True )

    model = His.omniHUXt_forecast(start_time, simtime = 27.27*u.day,
                                rmin = rmin, rmax = rmax,
                                dt_scale = dt_scale,
                                omni_input = omni_noicmes, buffertime = 0*u.day)
                                
    modelbg = His.omniHUXt_forecast(start_time, simtime = 27.27*u.day,
                                rmin = rmin, rmax = rmax,
                                dt_scale = dt_scale,
                                omni_input = omni_noicmes, buffertime = 0*u.day)
    
    #========================================================
    # Spheroidal cone cme

    cme = H.ConeCME(t_launch=0.0 * u.day,
                    longitude=onecme['lon'] * u.deg,
                    latitude=onecme['lat'] * u.deg,
                    initial_height=rmin,
                    width=2.0 * onecme['Ang_rad'] * u.deg,
                    v=onecme['CME_V']* (u.km / u.s),
                    thickness=0.0 * u.solRad,
                    cme_fixed_duration=True,
                    fixed_duration=11.5*60*60*u.s)
    
    modelbg.solve([])
    model.solve([cme])
        
    # The Earth time series can be plotted, along with OMNI data (downloaded on demand),using:
    fig, axs = plot_earth_timeseries(model, modelbg, plot_omni = True)
    axs[0].axvline(onecme['Disturbance_Time'], color='r')
    data_dir = project_dirs['figures']
    out_path = os.path.join(data_dir, "time_series_fcst")
    filename = f"{onecme['Disturbance_Time']}.pdf"
    filepath = os.path.join(out_path, filename)
    fig.savefig(filepath, bbox_inches='tight')
    #HA.animate(model, tag=filename, outputfilepath=filepath)
