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
import warnings
from datetime import timedelta
from matplotlib.pyplot import cm
# %%
md = os.path.dirname(os.path.dirname(os.getcwd())) + '\\input_data'
output_dir = '..\\..\\output_data'
# %%
ERAfiles = glob.glob(md + '\\ERA5\\ocean\\*.nc')
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
    #kwargs is a grid resolution of ref data either in km (km=(x,y))
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
    frcst, obs = dask.compute(frcst, obs)
    #print(location, 'line 1')
    bias = (frcst - obs).mean()
    #print(location, 'line 2')
    rmse = np.sqrt(((frcst - obs)**2).mean())
    #print(location, 'line 3')
    si = np.sqrt(
        (((frcst-frcst.mean())-(obs-obs.mean()))**2).mean()
        )/obs.mean()
    #print(location, 'line 4')
    cc = (
        ((frcst-frcst.mean())*(obs-obs.mean())).sum()/
        np.sqrt(
            (((frcst-frcst.mean())**2).sum()*((obs-obs.mean())**2).sum()))
        )
    #print(location, 'line 5')
    num_pairs = min(len(obs), len(frcst))
    #print(location, 'line 6')
    lsf = (
        ((obs**2).sum()-(obs.sum())**2/num_pairs)/
        ((obs*frcst).sum() - (obs.sum()*frcst.sum())/
        num_pairs)
    )
    return {'bias':bias, 'rmse':rmse, 'si':si, 'cc':cc, 'lsf':lsf}

# %%
def stats_table(obs_dict, ref_dict, var_obs,var_ref):
    #make a map for the var names which are different in era and buoys
    data_stats = pd.DataFrame(
        columns=['lat','lon', 'duration', 'records', 'bias', 
                'rmse', 'si', 'cc', 'lsf', 'min', 'max', 'meadian',
                'mean', 'missing'])
    locations = list(obs_dict.keys())
    for location in locations:
        
        #fix the observations which are not at round timesteps
        minutes=obs_dict[location].assign(
            minutes=obs_dict[location].index.minute.values
            ).minutes
        if len(obs_dict[location][(minutes!=0) & (minutes!=30)])>0:
            obs_dict[location].index = obs_dict[location].index.dt.round(freq='30min')

        #time-related stats
        full_series = pd.date_range(
            start=min(obs_dict[location].index), end=max(
                obs_dict[location].index), freq='H')
        duration = round_to(len(full_series)/8760, 0.1)
        frequency = (obs_dict[location].head().index[1]-obs_dict[location].head().index[0]).seconds
        if frequency/3600==1.0: #hourly frequency
            records = round_to(len(variable.dropna())/8760, 0.1)
        else: #half-hourly frequency
            records = round_to(len(variable.dropna())/17520, 0.1)
        missing = len(variable.dropna())/len(full_series)

        # calculate the main and additional stats
        variable = obs_dict[location].loc[:,var_obs]
        stats = calc_stats(variable, ref_dict[location].loc[:,var_ref])        
        
        min, max, median, mean = dask.compute(
            variable.min(), variable.max(), variable.median(), variable.mean())
        
        # add all stats as a table row
        row_data = [location[0], location[1], duration, records,
                    stats['bias'], stats['rmse'], stats['si'], stats['cc'],
                    stats['lsf'], min, max, median, mean, missing]
        
        data_stats = data_stats.append(
            pd.Series(row_data, index=data_stats.columns),
            ignore_index=True)

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
        #Try to fit the distribution
        try:
            #Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                
        
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
vars = ['VRM02','VRZA','VRPK','VGHS','VHM0','VAVH']
# %%
#file generator
def open_select_buoys(files, wave_vars):
    """
    the generator which checks the measurement duration and
    presence of required variables (wave_vars) and yields the datasets
    which have more than 1 year of data and at least 1 variable
    """
    less_than_year = []
    for file in files:
        xrds = xr.open_dataset(file)
        
        #skip the file with less than 1 year of measurements
        time_coverage = pd.to_datetime(
            xrds.time_coverage_end) - pd.to_datetime(xrds.time_coverage_start)
        if time_coverage < pd.Timedelta('365 days'):
            less_than_year.append(file.split('\\')[-1])
            continue
        
        # remove unwanted variables
        exclude_vars = []
        for var in list(xrds.keys()):
            if var not in wave_vars:
                exclude_vars.append(var)    
        xrds = xrds.drop_vars(exclude_vars)

        # skip the file if there're no variables left
        if not list(xrds.keys()):
            continue

        yield xrds
# %%
def organise_coords(dataset):
    """add lon-lat to coordinates and save their attributes"""
    lat_attr = dataset.LATITUDE.attrs
    lon_attr = dataset.LONGITUDE.attrs
    dataset.coords['LATITUDE'] = dataset.geospatial_lat_min
    dataset.coords['LONGITUDE'] = dataset.geospatial_lon_min
    dataset = dataset.expand_dims(['LATITUDE', 'LONGITUDE'])
    dataset.LATITUDE.attrs = lat_attr
    dataset.LONGITUDE.attrs = lon_attr
    return dataset
# %%
#works
dsets = []
for ds in tqdm(open_select_buoys(cmsfiles, vars), total=len(cmsfiles)):
    ds = organise_coords(ds)
    dsets.append(ds)
# %%

# %%

# %%
#create a dict to use with the previously defined functions
dds_dict = {}
not_in_dict = {}
for ds in dsets:
    lat = ds.LATITUDE
    lon = ds.LONGITUDE
    try:
        ds = ds.sel(DEPTH=0)
    except: 
        #print(f'location: {ds.platform_code}')
        not_in_dict[(float(lat),float(lon))] = ds
        continue
    
    dd = ds.sel(LATITUDE=lat, LONGITUDE=lon).to_dask_dataframe()
    dds_dict[(float(lat),float(lon))] = dd
#dsets[0].sel(LATITUDE='60.8833',LONGITUDE='20.7500', DEPTH=0)
#line above returns a ds 
# %%
for location, data in dds_dict.items():
    data = data.set_index('TIME')
    dds_dict[location] = data
# %%
#works but it compares lat and lon separately, hence it returns way too many 
# locations, potential to modify the func
cms_buoys = filter_buoys(dds_dict, 0.5, degrees=(0.5,0.5))
# %%

# %%
# %%
def create_era_dd(nc_reference, csv_buoys):
    #sampling should be in time format of python ('1d', '10y' etc)
    #nc_reference is a 3D netCDF file
    #csv_buoys 
    
    # the function is to select data from ERA5 file that matches files with buoy data
    # in terms of locations and time period. The data is then converted to pd.DataFrame
    reference_files = {}
    buoy_files = []
    for location, data in csv_buoys.items():
        #select netCDF part where the time and location 
        #corresponds to the observations
        nc_lat, nc_lon = round_to(location[0],0.5), round_to(location[1],0.5)
        #print(f'location: {location},line 154')
        try:
            nc_start, nc_end = data.index.min(), data.index.max()
         #   print(f'location: {location},line 157')
            time_query = (nc_reference.time >= nc_start) & (nc_reference.time <= nc_end)
        except:
            nc_start, nc_end = data.index.min().compute(), data.index.max().compute()
            time_query = (nc_reference.time >= nc_start) & (nc_reference.time <= nc_end)
        #print(f'location: {location},line 162')
        try:
            new_nc = nc_reference.sel(
                {'latitude':nc_lat, 'longitude':nc_lon}).where(
                    time_query, drop=True)
            buoy_files.append(location)
        except:
            continue
        #print(f'location: {location},line 621')
        #convert to pandas and save into dict
        csv_reference = new_nc.to_dataframe()
        reference_files[location] = csv_reference
        #print(f'{len(csv_buoys)-list(csv_buoys.keys()).index(location)-1} out of {len(csv_buoys)} files left')
    return reference_files, buoy_files
# %%
era_cms, locs = create_era_dd(era, cms_buoys)


# %%

# %%
all_locs = list(cms_buoys.keys())
for loc in all_locs:
    if loc not in locs:
        cms_buoys.pop(loc)
# %%
# the usual operations (e.g. in calc_stats) don't work 
# with vaex data, but do with dask 

# %%
for loc, data in cms_buoys.items():
    if loc in locs:
        print(loc, data.columns)
# %%
renamed = []
dropped = []
for loc, data in cms_buoys.items():
    if list(data.columns)[2:]==['VAVH']:
        data = data.rename(columns={'VAVH':'VHM0'})
        cms_buoys[loc] = data
        renamed.append(loc)
    if list(data.columns)[2:]==['VAVH', 'VHM0']:
        data = data.drop(columns=['VAVH'])
        dropped.append(loc)
    if list(data.columns)[2:]==['VGHS']:
        data = data.rename(columns={'VGHS':'VHM0'})
        cms_buoys[loc] = data
        renamed.append(loc)
# %%

#%%
#convert era to dask, otherwise calc_stats doesn't work
era_dds = {}
for loc,era_dataset in era_cms.items():
    era_dds[loc] = dask.dataframe.from_pandas(
        era_dataset,npartitions=24)
# %%
cms_stats = stats_table(cms_buoys,era_dds, 'VHM0', 'swh')
# %%
#code copied from above (to test before wrapping into a function)

# %%
#compare two distributions on ERA5 and buoy data on a plot + histogram
buoy_dists_dic = {}
era_dists_dic = {}
for location in locs[3:]:
    waves = pd.Series(cms_buoys[location].VHM0.dropna().values.compute())
    era_swh = pd.Series(era_dds[location].swh.dropna().values.compute())
    plt.figure(figsize=(12,8))
    ax = waves.plot(kind='hist',bins=25, density=True, color='peachpuff', label='buoy')
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
    plt.savefig(f'..\\..\\graphs\\era_vs_buoys\\glob_data\\{location}_dist.svg')
    buoy_dists_dic[location] = buoy_distr
    era_dists_dic[location] = era_distr
# %%


#%%
#calc stats in bins and temporarily
bins = list(np.arange(0, 10.5, 0.5))
subdfs =[]
long_locs = []
for location in locs:
    print(f'location {locs.index(location)} out of {len(locs)}')
    frequency = (cms_buoys[location].head(2).index[1]-cms_buoys[location].head(2).index[0]).seconds/3600
    cms_buoys[location] = cms_buoys[location].resample('1H').mean()
    binned_stats = pd.DataFrame(
        columns=['bias', 'rmse', 'si', 'cc', 'lsf', 'records'])
    for bin in bins:
        try:
            subset_buoys = cms_buoys[location][
                (cms_buoys[location].VHM0 >= bin) & (
                    cms_buoys[location].VHM0 < bins[bins.index(bin)+1])]
        except:
            subset_buoys = cms_buoys[location][
                (cms_buoys[location].VHM0 >= bin) & (cms_buoys[location].VHM0 < 10.5)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
        #timesteps = subset_buoys.index.values.compute()
        #if era_dds[location].index.min() > cms_buoys[location].index.min():

        #try:
        #    subset_era = era_dds[location].loc[timesteps]
        #except:
        #    subset_era = era_dds[location].loc[timesteps[1:]]
        #print(f'location: {location}, bin: {bin}, buoy length: {len(subset_buoys)}, era length: {len(subset_era)}')
        stat_dict = calc_stats(subset_buoys.VHM0, era_dds[location].swh)
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
        color = iter(cm.brg(np.linspace(0,1,len(binned_results.index.unique(level=1)))))
        for location, data in binned_results.groupby(level=[0,1]):
        #here location returns a tuple of lat and lon,
        #data is the corresponding dataframe
            #shade = next(color)
            ax[i,j].plot(bins, data[binned_results.columns[graphs]], linestyle='--')#, c=shade)#, color=colours[graphs])
            #condition = (cms_stats.lat==location[0]) & (cms_stats.lon==location[1])
            #num = cms_stats.loc[condition, 'records']
            #eq_years = float(num)
            #locs.append(f'({location[0]:.4f}, {location[1]:.4f}); {eq_years} eq.years')
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
fig.savefig(f'..\\..\\graphs\\era_vs_buoys\\BinnedStatsCMS.svg')
# %%
