import xarray as xr                                                                                                  
import dask                                                                                                
import dask.threaded                                                                                       
import dask.multiprocessing                                                                               
from dask.distributed import Client
import zarr                                                                               
import numpy as np 

import os

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2) 

ds=xr.open_mfdataset('/mnt/alberta/equipes/IGE/meom/workdir/albert/eORCA1/IPSLCM6ALR_eORCA1.2_mesh_mask.nc')
encoding = {vname: {'compressor': compressor} for vname in ds.variables}
ds.to_zarr(store='/mnt/alberta/equipes/IGE/meom/workdir/albert/eORCA1/zarr_IPSLCM6ALR_eORCA1.2_mesh_mask', encoding=encoding)
