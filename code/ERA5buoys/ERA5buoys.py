# %%
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import dtype
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import os
import glob
from basicfunc import round_to, proximity_calc, DM_to_DecDeg
#import info
import analysis
#from analysis import get_info_from_filename
#from analysis import arrange_files

# %%
md = os.path.dirname(os.path.dirname(os.getcwd())) + '\\input_data'

# %%
ERAfiles = glob.glob(md + '\\ERA5\\*.nc')
era = xr.open_mfdataset(ERAfiles)

# %%
column_names = {
    'Time (GMT)':'time',
    'Average (zero crossing) wave period (s)':'m0wp',
    'Significant wave height (m)':'swh',
    'Dominant (peak) wave direction (degrees)':'pwd',
    'Dominant (peak) wave period (s)':'pwp',
    'Maximum wave height (m)':'mwh',
    'Significant wave height (Hm0) (m)':'swh'}


# %%
def get_info_from_filename(filename):
    file_info = {}
    parts = filename.split('_') #getting <latN-lonW>, <start-end> etc
    #print(parts[2].split('-'))
    part1 = parts[2].split('-')[0]
    part2 = parts[2].split('-')[1]
    lat, Nhemi = part1[:-1], part1[-1]
    lon, Ehemi = part2[:-1], part2[-1]
    #get coordinates
    if Nhemi == 'N':
        DMlat = lat
    else: DMlat = '-'+lat
    if Ehemi == 'E':
        DMlon = lon
    else: DMlon = '-'+lon
    file_info['lat'] = DM_to_DecDeg(DMlat)
    file_info['lon'] = DM_to_DecDeg(DMlon)

    #get time period
    file_info['startday'] = parts[3].split('-')[0]
    file_info['endday'] = parts[3].split('-')[1]

    #get variables
    file_info['vars'] = parts[4].split('-')

    #get frequency
    file_info['frequency'] = int(parts[5].split('.')[0][:-1])
    return file_info
#get_info_from_filename(glob.glob(md + '\\buoys\\*.csv')[0])
# %%

# %%
def arrange_files(file_list, column_names):
    # columns is a dict for mapping
    csv_dfs = {}
    for file in file_list:
        csv = pd.read_csv(file)
        csv = csv.rename(columns=column_names)
        #convert time column to the same format as ERA5 and set it as index
        #(parse_dates and simpler time conversion didn't return expected
        # results, hence complicated conversion)
        csv.time = pd.to_datetime(csv.time)
        csv = csv.set_index('time')
        info = get_info_from_filename(file)
        #resample if required
        if (csv.index[1] - csv.index[0])== pd.Timedelta(1, unit='hour'):
            if csv.index[0].minute != 0:
                csv.index = csv.index - pd.Timedelta(csv.index[0].minute, 'T')
        else:
            csv = csv.resample('H').mean()
        csv_dfs[(info['lat'], info['lon'])] = csv
        print(f'{file_list.index(file)} out of {len(file_list)} files left')
    return csv_dfs
# %%
csv_list = glob.glob(md+'\\buoys' + '\*.csv')
buoy_data = arrange_files(csv_list, column_names)
# %%
def filter_buoys(buoy_dict, proximity, **kwargs):
    #kwargs is a grid resolution either in km (km=(x,y))
    # or in degrees. E.g. for ERA5 is degrees=(0.5, 0.5) 
    #proximity is as fraction relative to a grid point

    # the function is to read the files and
    # save those which are in proximity to reference dataset grid
    #horres = max(degrees)
    for key, value in kwargs.items():
        if key=='degrees':
            horres = max(value)
        elif key=='km':
            in_deg = round_to(value[0]/111,0.025), round_to(value[1]/111,0.025)
            horres = max(in_deg)
        
    # get latitude and longitude to check proximity to grid points
    buoys = {}
    dist_max = 0
    for location, data in buoy_dict.items():
        lat_round = round_to(location[0],horres)
        lon_round = round_to(location[1],horres)
        
        distance = np.sqrt((location[0]-lat_round)**2 +(location[1]-lon_round)**2)
        if distance > dist_max:
            dist_max = distance
        if distance <= proximity*horres:
            buoys[location]=data
    # calculate the geographical extent of the data
    west = round_to(min([key[0] for key in buoy_dict.keys()]), 0.001)
    north = round_to(max([key[1] for key in buoy_dict.keys()]), 0.001)
    east = round_to(max([key[0] for key in buoy_dict.keys()]), 0.001)
    south = round_to(min([key[1] for key in buoy_dict.keys()]), 0.001)
    print(f'data extent (WNES) is: {west, north, east, south}')
    #print(buoy_dict.keys())
    print(f'reference data grid: {kwargs.values()}')
    print(f'total files: {len(buoy_dict)}')
    print(f'accepted files (based on proximity to grid points): {len(buoys)}')
    print(f'largest distance from the grid point: {round_to(dist_max, 0.01)} degrees ({round_to(dist_max*111, 0.1)}km)')
    return buoys
# %%
def create_era_series(nc_reference, csv_buoys):
    #sampling should be in time format of python ('1d', '10y' etc)
    #nc_reference is a 3D netCDF file
    #csv_buoys 
    
    # the function is to select data from ERA5 file that matches files with buoy data
    # in terms of locations and time period. The data is then converted to pd.DataFrame
    reference_files = {}
    for location, data in csv_buoys.items():
        #select netCDF part where the time and location 
        #corresponds to the observations
        nc_lat, nc_lon = round_to(location[0],0.5), round_to(location[1],0.5)
        try:
            nc_start, nc_end = data.index[0], data.index[-1]
            time_query = (nc_reference.time >= nc_start) & (nc_reference.time <= nc_end)
        except:
            nc_start, nc_end = np.datetime64(data.index[0]), np.datetime64(data.index[1])
            time_query = (nc_reference.time >= nc_start) & (nc_reference.time <= nc_end)
        new_nc = nc_reference.sel(
            {'latitude':nc_lat, 'longitude':nc_lon}).where(
                time_query, drop=True)#.resample(
                    #{'time':sampling}).mean()
         
        #convert to pandas and save into dict
        csv_reference = new_nc.to_dataframe()
        reference_files[location] = csv_reference
    return reference_files
# %%
csv_buoys = filter_buoys(buoy_data, 0.25, degrees=(0.5,0.5))
# %%
era_dict = create_era_series(era, csv_buoys)
# %%
swh_era = era_dict[(51.5705,1.5787)].swh
swh_buoy = csv_buoys[(51.5705,1.5787)].swh
# %%
# %%

# %%
from distfit import distfit
dist = distfit()
dist.fit_transform(swh_buoy.dropna())
print(dist.summary)
# %%
dist.plot()
# %%
dist = distfit(distr='gamma')
dist.fit_transform(swh_buoy.dropna())
# %%
def calc_stats(obs,frcst):
    bias = (frcst - obs).mean()
    rmse = np.sqrt(((frcst - obs)**2).mean())
    re = (abs(frcst-obs)/obs*100).mean()
    si = np.sqrt(
        (((frcst-frcst.mean())-(obs-obs.mean()))**2).mean()
        )/obs.mean()
    cc = (
        ((frcst-frcst.mean())*(obs-obs.mean())).sum()/
        np.sqrt(
            (((frcst-frcst.mean())**2).sum()*((obs-obs.mean())**2).sum()))
        )
    num_pairs = min(len(obs), len(frcst))
    lsf = (
        ((obs**2).sum()-(obs.sum())**2/num_pairs)/
        ((obs*frcst).sum() - (obs.sum()*frcst.sum())/
        num_pairs)
    )
    return bias, rmse, re, si, cc, lsf
# %%

# %%
swh_era.hist()
# %%
swh_buoy.hist()
# %%
