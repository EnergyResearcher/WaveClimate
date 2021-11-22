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
    file_info['DMlat'] = DMlat
    file_info['DMlon'] = DMlon
    file_info['Declat'] = DM_to_DecDeg(DMlat)
    file_info['Declon'] = DM_to_DecDeg(DMlon)

    #get time period
    file_info['startday'] = parts[3].split('-')[0]
    file_info['endday'] = parts[3].split('-')[1]

    #get variables
    file_info['vars'] = parts[4].split('-')

    #get frequency
    file_info['frequency'] = int(parts[5].split('.')[0][:-1])
    return file_info
# %%
md = os.path.dirname(os.path.dirname(os.getcwd())) + '\\input_data'
csv_list = glob.glob(md+'\\buoys' + '\\*.csv')
# %%
coords = pd.DataFrame(
    columns=['DMlat','DMlon','Declat','Declon','file'])
DMlats = []
DMlons = []
Declats = []
Declons = []
filenames = []
for file in csv_list:
    info = get_info_from_filename(file)
    DMlats.append(info['DMlat'])
    DMlons.append(info['DMlon'])
    Declats.append(info['Declat'])
    Declons.append(info['Declon'])
    filenames.append(file.split('\\')[-1])
# %%
coords['DMlat'] = DMlats
coords['DMlon'] = DMlons
coords['Declat'] = Declats
coords['Declon'] = Declons
coords['file'] = filenames
# %%
coords.to_csv('buoy_coords.csv')
