import pandas as pd
from basicfunc import DM_to_DecDeg

def get_info_from_filename(filename):
    file_info = {}
    parts = filename.split('_')

    #get coordinates
    if parts[1].split('-')[0][-1] == 'N':
        DMlat = parts[1].split('-')[0][:-1]
    else: DMlat = f'-{parts[1].split("-")[0][:-1]}'
    if parts[1].split('-')[1][-1] == 'E':
        DMlon = parts[1].split("-")[1][:-1]
    else: DMlon = f'-{parts[1].split("-")[1][:-1]}'
    file_info['lat'] = DM_to_DecDeg(DMlat)
    file_info['lon'] = DM_to_DecDeg(DMlon)

    #get time period
    file_info['startday'] = parts[2].split('-')[0]
    file_info['endday'] = parts[2].split('-')[1]

    #get variables
    file_info['vars'] = parts[3].split('-')

    #get frequency
    file_info['frequency'] = int(parts[4].split('.')[0][:-1])
    return file_info

def csv_summary(csv_list):
    summary = pd.DataFrame(columns=['lat', 'lon', 'start', 'end', 'vars'])
    lats = []
    lons = []
    starts = []
    ends = []
    variables = []
    for file in csv_list:
        info = get_info_from_filename(file)
        lats.append(info['lat'])
        lons.append(info['lon'])
        starts.append(info['startday'])
        ends.append(info['endday'])
        variables.append(info['vars'])
    
    summary.lat = lats
    summary.lon = lons
    summary.start = starts
    summary.end = ends
    summary.vars = variables
    return summary

