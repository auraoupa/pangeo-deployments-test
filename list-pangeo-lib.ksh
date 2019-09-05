conda create --name pangeo-cloud -c conda-forge python=3.6.7 numpy=1.16.2 xarray=0.12.0 dask=1.1.4 matplotlib=3.0.3 cartopy=0.17.0 scipy=1.2.1 pandas=0.24.2 zarr=2.3.1 ipykernel numba jupyter

conda activate pangeo-cloud

python -m ipykernel install --user --name pangeo-cloud --display-name pangeo_cloud


#from pc to occigen
cd ~/anaconda2/pkgs
scp python-3.6.7-h381d211_1004.tar.bz2 numpy-1.16.2-py36h8b7e671_1.tar.bz2 xarray-0.12.0-py_0.tar.bz2 dask-1.1.4-py_0.tar.bz2 matplotlib-3.0.3-py36_1.tar.bz2 cartopy-0.17.0-py36h0aa2c8f_1004.tar.bz2 scipy-1.2.1-py36h09a28d5_1.tar.bz2 pandas-0.24.2-py36he6710b0_0.tar.bz2 zarr-2.3.1-py36_0.tar.bz2 ipykernel-5.1.0-py36h39e3cac_0.tar.bz2 numba-0.43.1-py36h962f231_0.tar.bz2 albert7a@occigen.cines.fr:/scratch/cnt0024/hmg2840/albert7a/anaconda2/pkgs/.

cd ~/Téléchargements
scp zarr-2.3.2-py36_0.tar.bz2 albert7a@occigen.cines.fr:/scratch/cnt0024/hmg2840/albert7a/anaconda2/pkgs/.

#on occigen
conda install --offline numpy-1.16.2-py36h8b7e671_1.tar.bz2 xarray-0.12.0-py_0.tar.bz2 dask-1.1.4-py_0.tar.bz2 matplotlib-3.0.3-py36_1.tar.bz2 cartopy-0.17.0-py36h0aa2c8f_1004.tar.bz2 scipy-1.2.1-py36h09a28d5_1.tar.bz2 pandas-0.24.2-py36he6710b0_0.tar.bz2 zarr-2.3.2-py36_0.tar.bz2 ipykernel-5.1.0-py36h39e3cac_0.tar.bz2 numba-0.43.1-py36h962f231_0.tar.bz2

pip install pandas
pip install zarr
