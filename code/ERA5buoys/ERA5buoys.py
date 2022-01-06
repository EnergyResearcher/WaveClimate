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
import scipy
import scipy.stats
#from analysis import get_info_from_filename
#from analysis import arrange_files
import dask
# %%
md = os.path.dirname(os.path.dirname(os.getcwd())) + '\\input_data'

# %%
ERAfiles = glob.glob(md + '\\ERA5\\*.nc')
era = xr.open_mfdataset(ERAfiles)
# %%

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
def arrange_files(file_list, column_names):
    # columns is a dict for mapping
    csv_dfs = {}
    for file in file_list:
        csv = pd.read_csv(file)
        csv = csv.rename(columns=column_names)
        #convert time column to the same format as ERA5 and set it as index
        #(parse_dates and simpler time conversion didn't return expected
        # results, hence complicated conversion)
        last = len(csv)
        #print(f'1. start: {csv.time[0]}, end: {csv.time[last-1]}')
        csv.time = pd.to_datetime(csv.time, format='%d/%m/%Y %H:%M')
        #print(f'2. start: {csv.time[0]}, end: {csv.time[last-1]}')
        csv = csv.set_index(csv.time)
        csv = csv.drop(columns=['time'])
        #print(f'3. start: {csv.index[0]}, end: {csv.index[-1]}')
        info = get_info_from_filename(file)
        #resample if required
        if (csv.index[1] - csv.index[0])== pd.Timedelta(1, unit='hour'):
            if csv.index[0].minute != 0:
                csv.index = csv.index - pd.Timedelta(csv.index[0].minute, 'T')
        else:
            csv = csv.resample('H').mean()
        #print(f'4. start: {csv.index[0]}, end: {csv.index[-1]}')
        csv_dfs[(info['lat'], info['lon'])] = csv[csv.index < pd.to_datetime('20210101')]
        print(f'{len(file_list)-file_list.index(file)} out of {len(file_list)} files left')
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
        print(f'{len(csv_buoys)-list(csv_buoys.keys()).index(location)} out of {len(csv_buoys)} files left')
    return reference_files
# %%
csv_buoys = filter_buoys(buoy_data, 0.5, degrees=(0.5,0.5))
# %%
era_dict = create_era_series(era, csv_buoys)
# %%
locations = list(csv_buoys.keys())
locations
# %%
def calc_stats(obs,frcst):
    bias = (frcst - obs).mean()
    rmse = np.sqrt(((frcst - obs)**2).mean())
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
    return {'bias':bias, 'rmse':rmse, 'si':si, 'cc':cc, 'lsf':lsf}

# %%
def stats_table(obs_dict, ref_dict, var):
    #make a map for the var names which are different in era and buoys
    data_stats = pd.DataFrame(
        columns=['lat','lon', 'duration', 'records', 'bias', 'rmse', 'si', 'cc', 'lsf'],
        index=range(len(obs_dict)))
    locations = list(obs_dict.keys())
    for location in locations:
        row = locations.index(location)
        stats = calc_stats(obs_dict[location][var], ref_dict[location][var])
        data_stats.loc[row, 'lat'] = location[0]
        data_stats.loc[row, 'lon'] = location[1]
        data_stats.loc[row, 'duration'] = round_to(len(pd.date_range(
            start=min(obs_dict[location].index), end=max(
                obs_dict[location].index), freq='D'))/365, 0.1)
        frequency = (obs_dict[location].index[1]-obs_dict[location].index[0]).seconds
        if frequency/3600==1.0:
            data_stats.loc[row, 'records'] = round_to(len(obs_dict[location][var].dropna())/8760, 0.1)
        else:
            data_stats.loc[row, 'records'] = round_to(len(obs_dict[location][var].dropna())/17520, 0.1)
    
        data_stats.loc[row, 'bias'] = stats['bias']
        data_stats.loc[row, 'rmse'] = stats['rmse']
        data_stats.loc[row, 'si'] = stats['si']
        data_stats.loc[row, 'cc'] = stats['cc']
        data_stats.loc[row, 'lsf'] = stats['lsf']
    return data_stats
# %%
#one file from the UK wavenet data has a nan column, this code is to remove it
csv_buoys[(51.879200000000004,1.488)].columns = ['m0wp', 'swh', 'nan']
csv_buoys[(51.879200000000004,1.488)] = csv_buoys[(51.879200000000004,1.488)].drop(columns = ['nan'])
# %%
selection = ['lognorm','exponweib']
#%%
stats = stats_table(csv_buoys, era_dict, 'swh')
#stats.to_csv('..\\..\\output_data\\swh_general_stats_v1.csv')
# %%

#%%
# code from stack overflow:
# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
from scipy.stats._continuous_distns import _distn_names
import warnings
#dstr_names = ['levy', 'wrapcauchy', 'weibull_min', 'rayleigh', 'lognorm','expon', 'pareto','gamma', 'genextreme', 'beta', 't']
# %%
#Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    #Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x+np.roll(x,-1))[:-1]/2.0
    
    #Best holders
    best_distributions = []

    #Estimate distribution parameters from data
    for ii, distribution in enumerate(
        [d for d in selection if not d in ['levy_stable','studentized_range']]):
        print("{:>3} / {:<3}: {}".format( ii+1, len(selection),distribution))

        distribution = getattr(scipy.stats, distribution)

        #Try to fit the distribution
        try:
            #Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                #fit dist to data
                params = distribution.fit(data)

                #Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                #Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y-pdf, 2.0))

                #if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf,x).plot(ax=ax)
                        end
                except Exception:
                    pass

                #identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        
        except Exception:
            pass
    return sorted(best_distributions, key=lambda x:x[2]) 
        
# %%
def make_pdf(dist, params, size=10000):
    """Generate distributions' Probability Distrubution Function"""

    #Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    #Get the start and end points of distribution
    if arg:
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale)
    else:
        start = dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, loc=loc, scale=scale)
    
    #Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf
# %%
# plotting 2 distributions with a histogram (working)
for location in locations:
    waves = pd.Series(csv_buoys[location].swh.dropna().values)
    fig, ax = plt.subplots(figsize=(12,8))
    waves.plot(axes=ax, kind='hist',bins=35, density=True, color='peachpuff')
    distributions = best_fit_distribution(waves, 200, ax)
    plt.legend(['lognorm','exponweib','buoy'])
    plt.savefig(f'..\\..\\graphs\\era_vs_buoys\\{location}_dist.svg')
    #dists_dic[location] = distributions
# %%
#compare two distributions on ERA5 and buoy data on a plot + histogram
buoy_dists_dic = {}
era_dists_dic = {}
for location in locations:
    waves = pd.Series(csv_buoys[location].swh.dropna().values)
    era_swh = pd.Series(era_dict[location].swh.dropna().values)
    plt.figure(figsize=(12,8))
    ax = waves.plot(kind='hist',bins=35, density=True, color='peachpuff', label='buoy')
    buoy_distr = best_fit_distribution(waves, 200)
    era_distr = best_fit_distribution(era_swh, 200)
    colors = ['limegreen','purple']
    #plot buoy distributions
    for bdistr in buoy_distr:
        ax = plt.plot(
            make_pdf(bdistr[0], bdistr[1]),
            color=colors[buoy_distr.index(bdistr)], linestyle='-',
            label='buoy:{}'.format(bdistr[0].name))

    #plot ERA5 distributions
    for edistr in era_distr:
        ax = plt.plot(
            make_pdf(edistr[0], edistr[1]),
            color=colors[era_distr.index(edistr)], linestyle='--',
            label='ERA:{}'.format(edistr[0].name))
    plt.title(f'Significant wave height PDF at {location}')
    plt.legend()
    plt.savefig(f'..\\..\\graphs\\era_vs_buoys\\{location}_dist.svg')
    buoy_dists_dic[location] = buoy_distr
    era_dists_dic[location] = era_distr
#%%
#code on the assumption that all distributions are used and compared in a 
# table using sse value (measure of how well a distr fits the data)
dfs = []
for locs, dists in dists_dic.items():
    distrs = []
    sse = []
    for distribution in dists:
        distrs.append(distribution[0].name)
        sse.append(distribution[2])
    loc_df = pd.DataFrame(data={'distr':distrs, 'sse':sse}) 
    loc_df.columns=pd.MultiIndex.from_product([[locs], ['distr','sse']])
    dfs.append(loc_df)
#%%
#save the data from above cell to a file
dfs1 = pd.concat(dfs, axis=1)
dfs1
dfs1.to_csv('distributions31.csv')
#%%
#calc stats in bins and temporarily
bins = list(np.arange(0, 10.5, 0.5))
subdfs =[]
long_locs = []
for location in locations:
    frequency = (csv_buoys[location].index[1]-csv_buoys[location].index[0]).seconds/3600
    binned_stats = pd.DataFrame(
        columns=['bias', 'rmse', 'si', 'cc', 'lsf', 'records'])
    for bin in bins:
        try:
            subset_buoys = csv_buoys[location][
                (csv_buoys[location].swh >= bin) & (
                    csv_buoys[location].swh < bins[bins.index(bin)+1])]
        except:
            subset_buoys = csv_buoys[location][
                (csv_buoys[location].swh >= bin) & (csv_buoys[location].swh < 10.5)]
        subset_era = era_dict[location][
            era_dict[location].index.isin(subset_buoys.index)]
        #print(f'location: {location}, bin: {bin}, buoy length: {len(subset_buoys)}, era length: {len(subset_era)}')
        stat_dict = calc_stats(subset_buoys.swh.values, subset_era.swh.values)
        statistic=pd.DataFrame(stat_dict, index=[bin])
        if frequency==1.0:
            statistic['records'] = round_to(len(subset_buoys)/24, 0.1)
        else: statistic['records'] = round_to(len(subset_buoys)/48, 0.1)
        binned_stats = pd.concat([binned_stats, statistic])
    if binned_stats['records'].sum() < 365:
        continue
    subdfs.append(binned_stats)
    long_locs.append(location)
binned_results = pd.concat(subdfs, keys=long_locs)
# %%
binned_results.to_csv('binned_swh.csv')
# %%
binned_results.index.set_names(['latitude', 'longitude', 'bin'], inplace=True)

# %%
# this and 2 cells below to plot the calculated stats for comparing ERA5
# and buoy data
from matplotlib.pyplot import cm
# %%
fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, figsize=(30,20))
#graphs = len(binned_results.columns)
i = 0
graphs = 0

while i < ax.shape[0]:
    j = 0
    while j < ax.shape[1]:
    #create plot for every metric and different lines for different locations
    # probably need to swap the for loops around
        locs = []
        color = iter(cm.brg(np.linspace(0,1,len(binned_results.index.unique(level=0)))))
        for location, data in binned_results.groupby(level=[0,1]):
        #here location returns a tuple of lat and lon,
        #data is the corresponding dataframe
            ax[i,j].plot(bins, data[binned_results.columns[graphs]], linestyle='--', c=next(color))#, color=colours[graphs])
            eq_years = float(stats.loc[stats.lat==location[0], 'records'])
            locs.append(f'({location[0]:.4f}, {location[1]:.4f}); {eq_years} eq.years')
        means = []
        for bin, df in binned_results.groupby(level=[2]):
            means.append(df[binned_results.columns[graphs]].mean())
        ax[i,j].plot(bins, means, linestyle='-', lw=6, color='slategray')
        ax[i,j].grid()
        ax[i,j].legend(locs+ ['mean'])
        ax[i,j].tick_params(axis='both', which='major', labelsize=25)
        ax[i,j].set_ylabel(binned_results.columns[graphs], fontsize=30)
        if i==1:
            ax[i,j].set_xlabel('Significant wave height, m', fontsize=30)
        j += 1
        graphs += 1
    i += 1
    
fig.tight_layout()

# %%
fig.savefig(f'..\\..\\graphs\\era_vs_buoys\\BinnedStats.svg')
# %%
cmsfiles = glob.glob(md + '\\cms\\PotDrifting=0\\*.nc')
cmsfiles.remove(md + '\\cms\\PotDrifting=0\\GL_WS_MO_44t14.nc')
cmsfiles.remove(md + '\\cms\\PotDrifting=0\\GL_WS_MO_45t01.nc')
cmsfiles.remove(md + '\\cms\\PotDrifting=0\\GL_WS_MO_46t29.nc')
# %%

# %%
from decimal import *
# %%
#works but crashes
buoy_dfs = {}
for ds in cms[:433]:
    lat = Decimal(ds.geospatial_lat_min)
    lon = Decimal(ds.geospatial_lon_min)
    print(ds.platform_code, f'({lat}, {lon})')
    try:
        df = ds.sel(
            POSITION=0, LATITUDE=lat, LONGITUDE=lon, DEPTH=0).drop_dims(
                ['LATITUDE', 'LONGITUDE']).to_dataframe()
    except:
        try:
        #print(ds.platform_code, f'({lat}, {lon})')
            df = ds.sel(
                POSITION=0, LATITUDE=lat, LONGITUDE=lon, DEPTH=0).to_dataframe()
        except: continue
    buoy_dfs[(lat, lon)] = df
    
# %%

# %%
#testing code? should work

ds1=xr.open_dataset(cmsfiles[0])
ds2=xr.open_dataset(cmsfiles[1])
exclude_vars = []
for var in list(ds1.keys()):
    if var not in vars:
        exclude_vars.append(var)    
ds1 = ds1.drop_vars(exclude_vars)
exclude_vars = []
for var in list(ds2.keys()):
    if var not in vars:
        exclude_vars.append(var)    
ds2 = ds2.drop_vars(exclude_vars)
xr.concat(
    [ds1.drop_dims(['LATITUDE', 'LONGITUDE']),
    ds2.drop_dims(['LATITUDE', 'LONGITUDE'])], dim=['TIME', 'DEPTH']).sel(concat_dim='TIME')
ds1.coords["LATITUDE"] = ds1.geospatial_lat_min
ds2.coords["LATITUDE"] = ds2.geospatial_lat_min
ds1.coords["LONGITUDE"] = ds1.geospatial_lon_min
ds2.coords["LONGITUDE"] = ds2.geospatial_lon_min
ds1 = ds1.expand_dims(['LATITUDE', 'LONGITUDE'])
ds2 = ds2.expand_dims(['LATITUDE', 'LONGITUDE'])
test = xr.concat([ds1, ds2], dim = ['TIME', 'DEPTH'])
test.sel(concat_dim=['TIME']).drop('concat_dim').squeeze('concat_dim')#=['DEPTH'])#.VHM0.mean(('LATITUDE', 'LONGITUDE'))
# %%

# %%
vars = ['VRM02','VRZA','VRPK','VGHS','VHM0']
# %%

# %%
#works
# read files, exclude undesired vars, add lon-lat to coordinates and
# save their attributes. Save all resulting datasets to list  
dsets = []
for file in cmsfiles:
    ds = xr.open_dataset(file)
    exclude_vars = []
    for var in list(ds.keys()):
        if var not in vars:
            exclude_vars.append(var)    
    ds = ds.drop_vars(exclude_vars)
    
    #print(f'location: {ds.platform_code}; {list(ds.keys())}')
    lat_attr = ds.LATITUDE.attrs
    lon_attr = ds.LONGITUDE.attrs
    ds.coords['LATITUDE'] = ds.geospatial_lat_min
    ds.coords['LONGITUDE'] = ds.geospatial_lon_min
    ds = ds.expand_dims(['LATITUDE', 'LONGITUDE'])
    ds.LATITUDE.attrs = lat_attr
    ds.LONGITUDE.attrs = lon_attr
    dsets.append(ds)
# %%
#doesn't work since the depth dim has different values
cms_full = xr.concat(dsets[:100], dim = ['TIME'])
# %%
ds_0depth = []
for ds in dsets:
    try:
        ds = ds.sel(DEPTH=0)
    except: continue
    ds_0depth.append(ds.VHM0)
# %%
#memory error - new error now for some reason
cms_full = xr.concat(ds_0depth[:2], dim = ['TIME'], fill_value=None)#, data_vars=['VHM0'])#,compat='override')

# %%

# %%
#trying to convert all files to a massive dask dataframe to then use
#as with the previous code
from dask.distributed import Client, progress
client = Client()
client.dashboard_link
# %%

# %%
#create a dict to use with the previously defined functions
# check the duration of the 
dds_dict = {}
not_in_dict = []
for ds in dsets:
    try:
        ds = ds.sel(DEPTH=0)
    except: 
        print(f'location: {ds.platform_code}')
        not_in_dict.append(ds.platform_code)
        continue
    lat = ds.LATITUDE
    lon = ds.LONGITUDE
    dd = ds.sel(LATITUDE=lat, LONGITUDE=lon).to_dask_dataframe()
    dds_dict[(float(lat),float(lon))] = dd
#dsets[0].sel(LATITUDE='60.8833',LONGITUDE='20.7500', DEPTH=0)
#line above returns a ds 
# %%
#works but it compares lat and lon separately, hence it returns way too many 
# locations, potential to modify the func
cms_buoys = filter_buoys(dds_dict, 0.5, degrees=(0.5,0.5))
# %%
#returns an error since index is not "indexable" (dd.index works, 
# dd.index[0] doesn't - use dd.loc instead)
era_cms = create_era_series(era, cms_buoys)
# %%
for location, data in cms_buoys.items():
    print(location)
    print(data.index[0])
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
        print(f'location: {location},line 154')
        try:
            nc_start, nc_end = data.index[0], data.index[-1]
            print(f'location: {location},line 157')
            time_query = (nc_reference.time >= nc_start) & (nc_reference.time <= nc_end)
        except:
            nc_start, nc_end = np.datetime64(data.index[0]), np.datetime64(data.index[1])
            time_query = (nc_reference.time >= nc_start) & (nc_reference.time <= nc_end)
        print(f'location: {location},line 162')
        new_nc = nc_reference.sel(
            {'latitude':nc_lat, 'longitude':nc_lon}).where(
                time_query, drop=True)#.resample(
                    #{'time':sampling}).mean()
         
        #convert to pandas and save into dict
        csv_reference = new_nc.to_dataframe()
        reference_files[location] = csv_reference
        print(f'{len(csv_buoys)-list(csv_buoys.keys()).index(location)} out of {len(csv_buoys)} files left')
    return reference_files