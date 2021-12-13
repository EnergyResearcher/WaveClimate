# %%
from ftplib import FTP
import pandas as pd
import xarray as xr
# %%
dir = 'C:\\Users\\649315\\OneDrive - hull.ac.uk\\PhD\\Academic\\WaveClimate\\input_data\\cms'
ftp = FTP('my.cmems-du.eu', user='vbessonova',passwd='detta-5-Ven')
ftp.cwd('Core/INSITU_GLO_WAVE_REP_OBSERVATIONS_013_045/history/MO')

# %%
buoy_db = pd.read_excel(dir + '\\CopernicusMarineService_buoyDB.xlsx')
file_list = ftp.nlst()


# %%
all_files0 = []
for f in file_list:
    print(f)
    buoy_id = f.split('_')[-1].split('.')[0]
    if ((buoy_db['Name']==buoy_id).sum()==0) and (buoy_id not in all_files0):
        #download file sending "RETR<name of file>" command
        # open(f, "w").write is executed after RETR succeeds and 
        # returns file binary data and writes it in the code directory
        ftp.retrbinary(f'RETR {f}', open(f, 'wb').write)
        all_files0.append(buoy_id)

# %%

# %%
ftp.quit()