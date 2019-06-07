#!/usr/bin/env python


import xarray as xr
import dask
import dask.threaded
import dask.multiprocessing
from dask.distributed import Client
import numpy as np                                                                                        
import zarr

c = Client()

import sys, glob
import numpy as np
import xarray as xr
import xscale.spectral.fft as xfft
import xscale 
import Wavenum_freq_spec_func as wfs
import time

zarru='/scratch/cnt0024/hmg2840/albert7a/NATL60/NATL60-CJM165-S/1h/SSU/zarr/NATL60-CJM165-SSU-1h-1m2deg2deg'
zarrv='/scratch/cnt0024/hmg2840/albert7a/NATL60/NATL60-CJM165-S/1h/SSV/zarr/NATL60-CJM165-SSV-1h-1m2deg2deg'
dsu=xr.open_zarr(zarru)
dsv=xr.open_zarr(zarrv)


lat=dsu['nav_lat']
lon=dsu['nav_lon']
 
latmin = 40.0; latmax = 45.0;
lonmin = -40.0; lonmax = -35.0;

domain = (lonmin<lon) * (lon<lonmax) * (latmin<lat) * (lat<latmax)
where = np.where(domain)

#get indice
jmin = np.min(where[0][:])
jmax = np.max(where[0][:])
imin = np.min(where[1][:])
imax = np.max(where[1][:])

latbox=lat[jmin:jmax,imin:imax]
lonbox=lon[jmin:jmax,imin:imax]



print('Select dates')
u_JFM=dsu.sel(time_counter=slice('2013-01-01','2013-03-31'))['vozocrtx']
v_JFM=dsv.sel(time_counter=slice('2013-01-01','2013-03-31'))['vomecrty']

print('Select box area')
u_JFM_box=u_JFM[:,jmin:jmax,imin:imax].chunk(chunks={'time_counter':-1,'x':-1,'y':-1})
v_JFM_box=v_JFM[:,jmin:jmax,imin:imax].chunk(chunks={'time_counter':-1,'x':-1,'y':-1})


# - get dx and dy
print('get dx and dy')
dx_JFM,dy_JFM = wfs.get_dx_dy(u_JFM_box[0],lonbox,latbox)


#... Detrend data in all dimension ...
print('Detrend data in all dimension')
u_JFM = wfs.detrendn(u_JFM_box,axes=[0,1,2])
v_JFM = wfs.detrendn(v_JFM_box,axes=[0,1,2])

#... Apply hanning windowing ...') 
print('Apply hanning windowing')
u_JFM = wfs.apply_window(u_JFM, u_JFM.dims, window_type='hanning')
v_JFM = wfs.apply_window(v_JFM, v_JFM.dims, window_type='hanning')

# - get derivatives
derivatives_JFM = wfs.velocity_derivatives(u_JFM, v_JFM, xdim='x', ydim='y', dx={'x': dx_JFM, 'y': dy_JFM})
dudx_JFM = derivatives_JFM['u_x']; dudy_JFM = derivatives_JFM['u_y']
dvdx_JFM = derivatives_JFM['v_x']; dvdy_JFM = derivatives_JFM['v_y']

# - compute terms
phi1_JFM = u_JFM*dudx_JFM + v_JFM*dudy_JFM
phi2_JFM = u_JFM*dvdx_JFM + v_JFM*dvdy_JFM

u_JFMhat = xfft.fft(u_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)
v_JFMhat = xfft.fft(v_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)

phi1_JFM_hat = xfft.fft(phi1_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)
phi2_JFM_hat = xfft.fft(phi2_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)

tm1_JFM = (u_JFMhat.conj())*phi1_JFM_hat
tm2_JFM = (v_JFMhat.conj())*phi2_JFM_hat

# - computer transfer
Nk_JFM,Nj_JFM,Ni_JFM = u_JFM.shape
transfer_2D_JFM = -1.0*(tm1_JFM + tm2_JFM)/np.square(Ni_JFM*Nj_JFM)
transfer_term_JFM = transfer_2D_JFM.real

#... Get frequency and wavenumber ... 
print('Get frequency and wavenumber')
ffrequency_JFM = u_JFMhat.f_time_counter
kx_JFM = u_JFMhat.f_x
ky_JFM = u_JFMhat.f_y

#... Get istropic wavenumber ... 
print('Get istropic wavenumber')
wavenumber_JFM,kradial_JFM = wfs.get_wavnum_kradial(kx_JFM,ky_JFM)

#... Get numpy array ... 
print('Get numpy array')
var_psd_np_JFM = transfer_term_JFM.values

#... Get 2D frequency-wavenumber field ... 
print('Get transfer')
transfer_JFM = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,var_psd_np_JFM) 

print('Get flux')
flux_JFM = wfs.get_flux_in_1D(kradial_JFM,wavenumber_JFM,var_psd_np_JFM)

ave to Netscdf file
# - build dataarray
print('Save to Netscdf file')
transfer_JFM_da = xr.DataArray(transfer_JFM,dims=['frequency','wavenumber'],name="transfer",coords=[ffrequency_JFM ,wavenumber_JFM])
flux_JFM_da = xr.DataArray(flux_JFM,dims=['frequency','wavenumber'],name="flux",coords=[ffrequency_JFM,wavenumber_JFM])
transfer_JFM_da.attrs['Name'] = '/scratch/cnt0024/hmg2840/albert7a/NATL60/NATL60-CJM165-S/1h/KE_Transfer_Flux_JFM_w_k_from_1h_NATL60-CJM165.nc'

transfer_JFM_da.to_dataset().to_netcdf(path='/scratch/cnt0024/hmg2840/albert7a/NATL60/NATL60-CJM165-S/1h/KE_Transfer_Flux_JFM_w_k_from_1h_NATL60-CJM165.nc',mode='w',engine='scipy')
flux_JFM_da.to_dataset().to_netcdf(path='/scratch/cnt0024/hmg2840/albert7a/NATL60/NATL60-CJM165-S/1h/KE_Transfer_Flux_JFM_w_k_from_1h_NATL60-CJM165.nc',mode='a',engine='scipy')

