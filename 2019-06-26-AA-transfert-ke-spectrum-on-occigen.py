

import xarray as xr
import dask
import dask.threaded
import dask.multiprocessing
from dask.distributed import Client
import numpy as np                                                                                        
import zarr

c = Client()




# In[2]:

print(np.__version__)


# In[19]:

c


# In[20]:

import sys, glob
import numpy as np
import xarray as xr
import xscale.spectral.fft as xfft
import xscale 
import Wavenum_freq_spec_func as wfs
import time



# In[21]:

get_ipython().magic(u'time')

zarru='/scratch/cnt0024/hmg2840/albert7a/eNATL60/zarr/eNATL60-BLBT02-SSU-1h'
zarrv='/store/albert7a/eNATL60/zarr/eNATL60-BLB002-SSV-1h'
dsu=xr.open_zarr(zarru)
dsv=xr.open_zarr(zarrv)




# In[22]:

dsu


# In[23]:

get_ipython().run_cell_magic(u'time', u'', u"lat=dsu['nav_lat']\nlon=dsu['nav_lon']\n \nlatmin = 40.0; latmax = 45.0;\nlonmin = -40.0; lonmax = -35.0;\n\ndomain = (lonmin<lon) * (lon<lonmax) * (latmin<lat) * (lat<latmax)\nwhere = np.where(domain)\n\n#get indice\njmin = np.min(where[0][:])\njmax = np.max(where[0][:])\nimin = np.min(where[1][:])\nimax = np.max(where[1][:])\n\nlatbox=lat[jmin:jmax,imin:imax]\nlonbox=lon[jmin:jmax,imin:imax]")


# In[24]:

print(jmin,jmax,imin,imax)


# In[25]:

get_ipython().run_cell_magic(u'time', u'', u"\nprint('Select dates')\nu_JFM=dsu.sel(time_counter=slice('2010-01-01','2010-03-31'))['sozocrtx']\nv_JFM=dsv.sel(time_counter=slice('2010-01-01','2010-03-31'))['somecrty']\n\n\nprint('Select box area')\nu_JFM_box=u_JFM[:,jmin:jmax,imin:imax].chunk({'time_counter':10,'x':120,'y':120})\nv_JFM_box=v_JFM[:,jmin:jmax,imin:imax].chunk({'time_counter':10,'x':120,'y':120})\n\n\n\n# - get dx and dy\nprint('get dx and dy')\ndx_JFM,dy_JFM = wfs.get_dx_dy(u_JFM_box[0],lonbox,latbox)\n\n\n#... Detrend data in all dimension ...\nprint('Detrend data in all dimension')\nu_JFM = wfs.detrendn(u_JFM_box,axes=[0,1,2])\nv_JFM = wfs.detrendn(v_JFM_box,axes=[0,1,2])\n\n#... Apply hanning windowing ...') \nprint('Apply hanning windowing')\nu_JFM = wfs.apply_window(u_JFM, u_JFM.dims, window_type='hanning')\nv_JFM = wfs.apply_window(v_JFM, v_JFM.dims, window_type='hanning')\n\n\n# - get derivatives\nderivatives_JFM = wfs.velocity_derivatives(u_JFM, v_JFM, xdim='x', ydim='y', dx={'x': dx_JFM, 'y': dy_JFM})\ndudx_JFM = derivatives_JFM['u_x']; dudy_JFM = derivatives_JFM['u_y']\ndvdx_JFM = derivatives_JFM['v_x']; dvdy_JFM = derivatives_JFM['v_y']\n\n# - compute terms\nphi1_JFM = u_JFM*dudx_JFM + v_JFM*dudy_JFM\nphi2_JFM = u_JFM*dvdx_JFM + v_JFM*dvdy_JFM\n\nu_JFMhat = xfft.fft(u_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)\nv_JFMhat = xfft.fft(v_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)\n\nphi1_JFM_hat = xfft.fft(phi1_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)\nphi2_JFM_hat = xfft.fft(phi2_JFM, dim=('time_counter', 'x', 'y'), dx={'x': dx_JFM, 'y': dx_JFM}, sym=True)\n\ntm1_JFM = (u_JFMhat.conj())*phi1_JFM_hat\ntm2_JFM = (v_JFMhat.conj())*phi2_JFM_hat\n\n# - computer transfer\nNk_JFM,Nj_JFM,Ni_JFM = u_JFM.shape\ntransfer_2D_JFM = -1.0*(tm1_JFM + tm2_JFM)/np.square(Ni_JFM*Nj_JFM)\ntransfer_term_JFM = transfer_2D_JFM.real\n\n#... Get frequency and wavenumber ... \nprint('Get frequency and wavenumber')\nffrequency_JFM = u_JFMhat.f_time_counter\nkx_JFM = u_JFMhat.f_x\nky_JFM = u_JFMhat.f_y\n\n#... Get istropic wavenumber ... \nprint('Get istropic wavenumber')\nwavenumber_JFM,kradial_JFM = wfs.get_wavnum_kradial(kx_JFM,ky_JFM)\n")


# In[26]:

#... Get numpy array ... 
print('Get numpy array')
var_psd_np_JFM = transfer_term_JFM.values


# In[ ]:


#... Get 2D frequency-wavenumber field ... 
print('Get transfer')
transfer_JFM = wfs.get_f_k_in_2D(kradial_JFM,wavenumber_JFM,var_psd_np_JFM) 

print('Get flux')
flux_JFM = wfs.get_flux_in_1D(kradial_JFM,wavenumber_JFM,var_psd_np_JFM)

