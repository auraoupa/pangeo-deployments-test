
import xarray as xr
import dask
import dask.threaded
import dask.multiprocessing
from dask.distributed import Client
import numpy as np                                                                                        
import zarr

c = Client()


c


import sys, glob
import numpy as np
import xarray as xr
import xscale.spectral.fft as xfft
import xscale 
import Wavenum_freq_spec_func as wfs
import time



filesu='/store/albert7a/NATL60/NATL60-CJM165-S/1h/surf-U-V/NATL60-CJM165_y201?m??d??.1h_SSU.nc'
filesv='/store/albert7a/NATL60/NATL60-CJM165-S/1h/surf-U-V/NATL60-CJM165_y201?m??d??.1h_SSV.nc'

dsu=xr.open_mfdataset(filesu, parallel=True, concat_dim='time_counter',chunks={'time_counter':672,'y':120,'x':120})
dsv=xr.open_mfdataset(filesv, parallel=True, concat_dim='time_counter',chunks={'time_counter':672,'y':120,'x':120})


lat=dsu['nav_lat']
lon=dsu['nav_lon']
 
latmin = 40.0; latmax = 45.0;
lonmin = -40.0; lonmax = -35.0;

domain = (lonmin<lon) * (lon<lonmax) * (latmin<lat) * (lat<latmax)
where = np.where(domain)

jmin = np.min(where[0][:])
jmax = np.max(where[0][:])
imin = np.min(where[1][:])
imax = np.max(where[1][:])

latbox=lat[jmin:jmax,imin:imax]
lonbox=lon[jmin:jmax,imin:imax]


#print('Select dates')
#u_JFM=dsu.sel(time_counter=slice('2013-01-01','2013-03-31'))['vozocrtx']
#v_JFM=dsv.sel(time_counter=slice('2013-01-01','2013-03-31'))['vomecrty']
#
#print('Select box area')
#u_JFM_box=u_JFM[:,jmin:jmax,imin:imax]
#v_JFM_box=v_JFM[:,jmin:jmax,imin:imax]
#
#
## - get dx and dy
#print('get dx and dy')
#dx_JFM,dy_JFM = wfs.get_dx_dy(u_JFM_box[0],lonbox,latbox)
#
#
##... Detrend data in all dimension ...
#print('Detrend data in all dimension')
#u_JFM = wfs.detrendn(u_JFM_box,axes=[0,1,2])
#v_JFM = wfs.detrendn(v_JFM_box,axes=[0,1,2])
#
##... Apply hanning windowing ...') 
#print('Apply hanning windowing')
#u_JFM = wfs.apply_window(u_JFM, u_JFM.dims, window_type='hanning')
#v_JFM = wfs.apply_window(v_JFM, v_JFM.dims, window_type='hanning')
#
#
##... Apply hanning windowing ...') 
#print('FFT ')
#u_JFMhat = xfft.fft(u_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)
#v_JFMhat = xfft.fft(v_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)
#
##... Apply hanning windowing ...') 
#print('PSD ')
#u_JFM_psd = xfft.psd(u_JFMhat)
#v_JFM_psd = xfft.psd(v_JFMhat)
#
#
##... Get frequency and wavenumber ... 
#print('Get frequency and wavenumber')
#frequency_JFM = u_JFMhat.f_time_counter
#kx_JFM = u_JFMhat.f_x
#ky_JFM = u_JFMhat.f_y
#
##... Get istropic wavenumber ... 
#print('Get istropic wavenumber')
#wavenumber_JFM,kradial_JFM = wfs.get_wavnum_kradial(kx_JFM,ky_JFM)
#
##... Get numpy array ... 
#print('Get numpy array')
#u_JFM_psd_np = u_JFM_psd.values
#v_JFM_psd_np = v_JFM_psd.values
#
##... Get 2D frequency-wavenumber field ... 
#print('Get f k in 2D')
#u_JFM_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,u_JFM_psd_np)
#v_JFM_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,v_JFM_psd_np)
#
#KE_JFM_wavenum_freq_spectrum=0.5*(u_JFM_wavenum_freq_spectrum+v_JFM_wavenum_freq_spectrum)
#

# Save to Netscdf file
# - build dataarray
#print('Save to Netscdf file')
#KE_JFM_wavenum_freq_spectrum_da = xr.DataArray(KE_JFM_wavenum_freq_spectrum,dims=['frequency','wavenumber'],name="Ke_spectrum",coords=[frequency_JFM ,wavenumber_JFM])
#KE_JFM_wavenum_freq_spectrum_da.attrs['Name'] = 'KE_Spectrum_JFM_w_k_from_1h_NATL60-CJM165.nc'

#KE_JFM_wavenum_freq_spectrum_da.to_dataset().to_netcdf(path='/mnt/alberta/equipes/IGE/meom/workdir/albert/NATL60/NATL60-CJM165-S/1h/KE_Spectrum_JFM_w_k_from_1h_NATL60-CJM165.nc',mode='w',engine='scipy')

print('Select dates')
u_JAS=dsu.sel(time_counter=slice('2013-07-01','2013-09-29'))['vozocrtx']
v_JAS=dsv.sel(time_counter=slice('2013-07-01','2013-09-29'))['vomecrty']

print('Select box area')
u_JAS_box=u_JAS[:,jmin:jmax,imin:imax].chunk({'time_counter':-1,'x':120,'y':120})
v_JAS_box=v_JAS[:,jmin:jmax,imin:imax].chunk({'time_counter':-1,'x':120,'y':120})


# - get dx and dy
print('get dx and dy')
dx_JAS,dy_JAS = wfs.get_dx_dy(u_JAS_box[0],lonbox,latbox)


#... Detrend data in all dimension ...
print('Detrend data in all dimension')
u_JAS = wfs.detrendn(u_JAS_box,axes=[0,1,2])
v_JAS = wfs.detrendn(v_JAS_box,axes=[0,1,2])

#... Apply hanning windowing ...') 
print('Apply hanning windowing')
u_JAS = wfs.apply_window(u_JAS, u_JAS.dims, window_type='hanning')
v_JAS = wfs.apply_window(v_JAS, v_JAS.dims, window_type='hanning')


#... Apply hanning windowing ...') 
print('FFT ')
u_JAShat = xfft.fft(u_JAS, dim=('time_counter', 'x', 'y'), dx={'x': dx_JAS, 'y': dx_JAS}, sym=True)
v_JAShat = xfft.fft(v_JAS, dim=('time_counter', 'x', 'y'), dx={'x': dx_JAS, 'y': dx_JAS}, sym=True)

#... Apply hanning windowing ...') 
print('PSD ')
u_JAS_psd = xfft.psd(u_JAShat)
v_JAS_psd = xfft.psd(v_JAShat)


#... Get frequency and wavenumber ... 
print('Get frequency and wavenumber')
frequency_JAS = u_JAShat.f_time_counter
kx_JAS = u_JAShat.f_x
ky_JAS = u_JAShat.f_y

#... Get istropic wavenumber ... 
print('Get istropic wavenumber')
wavenumber_JAS,kradial_JAS = wfs.get_wavnum_kradial(kx_JAS,ky_JAS)

#... Get numpy array ... 
print('Get numpy array')
u_JAS_psd_np = u_JAS_psd.values
v_JAS_psd_np = v_JAS_psd.values

#... Get 2D frequency-wavenumber field ... 
print('Get f k in 2D')
u_JAS_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JAS,wavenumber_JAS,u_JAS_psd_np)
v_JAS_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial_JAS,wavenumber_JAS,v_JAS_psd_np)

KE_JAS_wavenum_freq_spectrum=0.5*(u_JAS_wavenum_freq_spectrum+v_JAS_wavenum_freq_spectrum)


# In[8]:


# Save to Netscdf file
# - build dataarray
print('Save to Netscdf file')
KE_JAS_wavenum_freq_spectrum_da = xr.DataArray(KE_JAS_wavenum_freq_spectrum,dims=['frequency','wavenumber'],name="Ke_spectrum",coords=[frequency_JAS ,wavenumber_JAS])
KE_JAS_wavenum_freq_spectrum_da.attrs['Name'] = 'KE_Spectrum_JAS_w_k_from_1h_NATL60-CJM165.nc'

KE_JAS_wavenum_freq_spectrum_da.to_dataset().to_netcdf(path='/mnt/alberta/equipes/IGE/meom/workdir/albert/NATL60/NATL60-CJM165-S/1h/KE_Spectrum_JAS_w_k_from_1h_NATL60-CJM165.nc',mode='w',engine='scipy')


# In[ ]:




