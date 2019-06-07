conda create --name pangeo-cloud -c conda-forge python=3.6.7 numpy=1.16.2 xarray=0.12.0 dask=1.1.4 matplotlib=3.0.3 cartopy=0.17.0 scipy=1.2.1 pandas=0.24.2 zarr=2.3.1 ipykernel numba

conda activate pangeo-cloud

python -m ipykernel install --user --name pangeo-cloud --display-name pangeo_cloud
