#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import dask
import dask.threaded
import dask.multiprocessing
from dask.distributed import Client
import numpy as np                                                                                        
import zarr

c = Client()


# In[2]:


c


# In[3]:


import sys, glob
import numpy as np
import xarray as xr
import xscale.spectral.fft as xfft
import xscale 
import Wavenum_freq_spec_func as wfs
import time


# In[4]:


import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib.colors import LogNorm

seq_cmap = mplcm.Blues
div_cmap = mplcm.seismic


# In[5]:


zarru='/mnt/alberta/equipes/IGE/meom/workdir/albert/NATL60/NATL60-CJM165-S/1h/SSU/zarr/NATL60-CJM165-SSU-1h-1m2deg2deg'
zarrv='/mnt/alberta/equipes/IGE/meom/workdir/albert/NATL60/NATL60-CJM165-S/1h/SSV/zarr/NATL60-CJM165-SSV-1h-1m2deg2deg'
dsu=xr.open_zarr(zarru)
dsv=xr.open_zarr(zarrv)


# In[6]:


get_ipython().run_cell_magic(u'time', u'', u"lat=dsu['nav_lat']\nlon=dsu['nav_lon']\n \nlatmin = 40.0; latmax = 45.0;\nlonmin = -40.0; lonmax = -35.0;\n\ndomain = (lonmin<lon) * (lon<lonmax) * (latmin<lat) * (lat<latmax)\nwhere = np.where(domain)\n\n#get indice\njmin = np.min(where[0][:])\njmax = np.max(where[0][:])\nimin = np.min(where[1][:])\nimax = np.max(where[1][:])\n\nlatbox=lat[jmin:jmax,imin:imax]\nlonbox=lon[jmin:jmax,imin:imax]")


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u"\nprint('Select dates')\nu_JFM=dsu.sel(time_counter=slice('2013-01-01','2013-03-31'))['vozocrtx']\nv_JFM=dsv.sel(time_counter=slice('2013-01-01','2013-03-31'))['vomecrty']\n\nprint('Select box area')\nu_JFM_box=u_JFM[:,jmin:jmax,imin:imax]\nv_JFM_box=v_JFM[:,jmin:jmax,imin:imax]\n\n# - remove NaN\nprint('rechunk')\nu_JFM_rechunk=u_JFM_box.chunk(chunks={'time_counter':1,'y':-1})\nv_JFM_rechunk=v_JFM_box.chunk(chunks={'time_counter':1,'y':-1})\n\nprint('Interpolate')\nu_JFM = u_JFM_rechunk.interpolate_na(dim='y')\nv_JFM = v_JFM_rechunk.interpolate_na(dim='y')\n\n# - get dx and dy\nprint('get dx and dy')\ndx_JFM,dy_JFM = wfs.get_dx_dy(u_JFM[0],lonbox,latbox)\n\n\n#... Detrend data in all dimension ...\nprint('Detrend data in all dimension')\nu_JFM = wfs.detrendn(u_JFM,axes=[0,1,2])\nv_JFM = wfs.detrendn(v_JFM,axes=[0,1,2])\n\n#... Apply hanning windowing ...') \nprint('Apply hanning windowing')\nu_JFM = wfs.apply_window(u_JFM, u_JFM.dims, window_type='hanning')\nv_JFM = wfs.apply_window(v_JFM, v_JFM.dims, window_type='hanning')\n\n# - get derivatives\nderivatives_JFM = wfs.velocity_derivatives(u_JFM, v_JFM, xdim='x', ydim='y', dx={'x': dx_JFM, 'y': dy_JFM})\ndudx_JFM = derivatives_JFM['u_x']; dudy_JFM = derivatives_JFM['u_y']\ndvdx_JFM = derivatives_JFM['v_x']; dvdy_JFM = derivatives_JFM['v_y']\n\n# - compute terms\nphi1_JFM = u_JFM*dudx_JFM + v_JFM*dudy_JFM\nphi2_JFM = u_JFM*dvdx_JFM + v_JFM*dvdy_JFM\n\nu_JFMhat = xfft.fft(u_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)\nv_JFMhat = xfft.fft(v_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)\n\nphi1_JFM_hat = xfft.fft(phi1_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)\nphi2_JFM_hat = xfft.fft(phi2_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)\n\ntm1_JFM = (u_JFMhat.conj())*phi1_JFM_hat\ntm2_JFM = (v_JFMhat.conj())*phi2_JFM_hat\n\n# - computer transfer\nNk_JFM,Nj_JFM,Ni_JFM = u_JFM.shape\ntransfer_2D_JFM = -1.0*(tm1_JFM + tm2_JFM)/np.square(Ni_JFM*Nj_JFM)\ntransfer_term_JFM = transfer_2D_JFM.real\n\n#... Get frequency and wavenumber ... \nprint('Get frequency and wavenumber')\nffrequency_JFM = u_JFMhat.f_time_counter\nkx_JFM = u_JFMhat.f_x\nky_JFM = u_JFMhat.f_y\n\n#... Get istropic wavenumber ... \nprint('Get istropic wavenumber')\nwavenumber_JFM,kradial_JFM = wfs.get_wavnum_kradial(kx_JFM,ky_JFM)\n\n#... Get numpy array ... \nprint('Get numpy array')\nvar_psd_np_JFM = transfer_term_JFM.values\n\n#... Get 2D frequency-wavenumber field ... \nprint('Get transfer')\ntransfer_JFM = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,var_psd_np_JFM) \n\nprint('Get flux')\nflux_JFM = wfs.get_flux_in_1D(kradial_JFM,wavenumber_JFM,var_psd_np_JFM)")


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u"u_JAS=dsu.sel(time_counter=slice('2013-06-01','2013-09-30'))['vozocrtx']\nv_JAS=dsv.sel(time_counter=slice('2013-06-01','2013-09-30'))['vomecrty']\n\nu_JAS_box=u_JAS.where((lonmin<lon) & (lon<lonmax) & (latmin<lat) & (lat<latmax),drop=True)\nv_JAS_box=v_JAS.where((lonmin<lon) & (lon<lonmax) & (latmin<lat) & (lat<latmax),drop=True)\n\n# - remove NaN\nu_JAS = u_JAS_box.interpolate_na(dim='y')\nv_JAS = v_JAS_box.interpolate_na(dim='x')\n\n# - get dx and dy\ndx_JAS,dy_JAS = wfs.get_dx_dy(u_JAS[0],lonbox,latbox)\n\n\n#... Detrend data in all dimension ...\nprint('Detrend data in all dimension')\nu_JAS = wfs.detrendn(u_JAS,axes=[0,1,2])\nv_JAS = wfs.detrendn(v_JAS,axes=[0,1,2])\n\n#... Apply hanning windowing ...') \nprint('Apply hanning windowing')\nu_JAS = wfs.apply_window(u_JAS, u_JAS.dims, window_type='hanning')\nv_JAS = wfs.apply_window(v_JAS, v_JAS.dims, window_type='hanning')\n\n# - get derivatives\nderivatives_JAS = wfs.velocity_derivatives(u_JAS, v_JAS, xdim='x', ydim='y', dx={'x': dx_JAS, 'y': dy_JAS})\ndudx_JAS = derivatives_JAS['u_x']; dudy_JAS = derivatives_JAS['u_y']\ndvdx_JAS = derivatives_JAS['v_x']; dvdy_JAS = derivatives_JAS['v_y']\n\n# - compute terms\nphi1_JAS = u_JAS*dudx_JAS + v_JAS*dudy_JAS\nphi2_JAS = u_JAS*dvdx_JAS + v_JAS*dvdy_JAS\n\nu_JAShat = xfft.fft(u_JAS, dim=('time_counter', 'x', 'y'), dx={'x': dx_JAS, 'y': dx_JAS}, sym=True)\nv_JAShat = xfft.fft(v_JAS, dim=('time_counter', 'x', 'y'), dx={'x': dx_JAS, 'y': dx_JAS}, sym=True)\n\nphi1_JAS_hat = xfft.fft(phi1_JAS, dim=('time_counter', 'x', 'y'), dx={'x': dx_JAS, 'y': dx_JAS}, sym=True)\nphi2_JAS_hat = xfft.fft(phi2_JAS, dim=('time_counter', 'x', 'y'), dx={'x': dx_JAS, 'y': dx_JAS}, sym=True)\n\ntm1_JAS = (u_JAShat.conj())*phi1_JAS_hat\ntm2_JAS = (v_JAShat.conj())*phi2_JAS_hat\n\n# - computer transfer\nNk_JAS,Nj_JAS,Ni_JAS = u_JAS.shape\ntransfer_2D_JAS = -1.0*(tm1_JAS + tm2_JAS)/np.square(Ni_JAS*Nj_JAS)\ntransfer_term_JAS = transfer_2D_JAS.real\n\n#... Get frequency and wavenumber ... \nprint('Get frequency and wavenumber')\nffrequency_JAS = u_JAShat.f_time\nkx_JAS = u_JAShat.f_x\nky_JAS = u_JAShat.f_y\n\n#... Get istropic wavenumber ... \nprint('Get istropic wavenumber')\nwavenumber_JAS,kradial_JAS = wfs.get_wavnum_kradial(kx_JAS,ky_JAS)\n\n#... Get numpy array ... \nprint('Get numpy array')\nvar_psd_np_JAS = transfer_term_JAS.values\n\n#... Get 2D frequency-wavenumber field ... \nprint('Get transfer')\ntransfer_JAS = wfs.get_f_k_in_2D(kradial_JAS,wavenumber_JAS,var_psd_np_JAS) \n\nprint('Get flux')\nflux_JAS = wfs.get_flux_in_1D(kradial_JAS,wavenumber_JAS,var_psd_np_JAS)")


# In[ ]:


sec_to_hour = 3600.0

cmap = 'bwr'

fig=plt.figure(figsize=(30,20))

ax = plt.subplot(121)
plt.pcolormesh(wavenumber_JFM,sec_to_hour*ffrequency_JFM,10*flux_JFM,cmap=cmap,vmin=-0.1,vmax=0.1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
ax.set_ylabel('frequency (cph)',fontsize=15)
ax.set_xlim(wavenumber_JFM.min(),wavenumber_JFM.max())
ax.set_ylim(1E-5,8E-1)
ax.set_title('KE transfer flux JFM Small Box 1h',size=18)
ax.tick_params(labelsize=15)
plt.legend
plt.colorbar()

ax = plt.subplot(122)
plt.pcolormesh(wavenumber_JAS,sec_to_hour*ffrequency_JAS,10*flux_JAS,cmap=cmap,vmin=-0.1,vmax=0.1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('wavenumber (cpkm)',fontsize=15)
ax.set_ylabel('frequency (cph)',fontsize=15)
ax.set_xlim(wavenumber_JAS.min(),wavenumber_JAS.max())
ax.set_ylim(1E-5,8E-1)
ax.set_title('KE transfer flux JAS Small Box 1h',size=18)
ax.tick_params(labelsize=15)
plt.legend
plt.colorbar()

plt.title('run NATL60, year 2012, box : 40-45°Nx40-35°W')
plt.savefig('KE_transfer_flux_llc4320_JFM-JAS_smallbox.png')


# In[ ]:




