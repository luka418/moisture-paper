# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:51:41 2023

@author: fanwang
"""

import numpy as np
import scipy.io as scio
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
from mpl_toolkits.axes_grid1 import AxesGrid
import cmaps
import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

plt.rcParams['font.sans-serif'] ="Arial"
import warnings
warnings.filterwarnings('ignore')

def add_Chinese_provinces(ax, **kwargs):
    proj = ccrs.PlateCarree()
    shp_filepath = 'D:/Python Programes/china_shp/bou2_4p.shp'
    reader = Reader(shp_filepath)
    provinces = cfeature.ShapelyFeature(reader.geometries(), proj)
    ax.add_feature(provinces, **kwargs)
    
def plot_china(lat1, lon1, data, vmin, vmax):  
    fig = plt.figure(figsize=(20, 10), dpi=100)
    #plt.subplots_adjust(left=0.05, bottom=0.2, right=0.99, top=0.97, hspace=0.2, wspace=0.2)
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(1, 3),
                    axes_pad=0.7,
                    cbar_location='right',
                    cbar_mode='single',#'single',
                    cbar_pad=0.1,
                    cbar_size='2%',
                    label_mode='')  # note the empty label_mode
    for i, ax in enumerate(axgr):
            sf = shapefile.Reader(r"D:/Python Programes/shp_data/country1")
            for shape_rec in sf.shapeRecords():
                if shape_rec.record[2] == 'China':#Hunan Sheng
                    vertices = []
                    codes = []
                    pts = shape_rec.shape.points
                    prt = list(shape_rec.shape.parts) + [len(pts)]
                    for ii in range(len(prt) - 1):
                        for jj in range(prt[ii], prt[ii+1]):
                            vertices.append((pts[jj][0], pts[jj][1]))
                        codes += [Path.MOVETO]
                        codes += [Path.LINETO] * (prt[ii+1] - prt[ii] -2)
                        codes += [Path.CLOSEPOLY]
                    clip = Path(vertices, codes)
                    clip = PathPatch(clip, transform=ax.transData)
            p = ax.contourf(
                lon1, lat1, data[i],
                levels=np.linspace(vmin, vmax, 31), 
                cmap=cmaps.WhBlGrYeRe,
                extend='both'
                )

            for contour in p.collections:
                contour.set_clip_path(clip)
            axgr.cbar_axes[i].colorbar(p)
            add_Chinese_provinces(ax, lw=0.5, ec='k', fc='none')
            ax.coastlines(resolution='10m', lw=0.3)
            
            ax.set_xlim(70, 137)
            ax.set_xticks(np.linspace(70, 130, 4), crs=projection)
            ax.set_ylim(15, 55)
            ax.set_yticks(np.linspace(15, 55, 3), crs=projection)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            #ax.text(180,90,"%.2f" %names[i], va='bottom', ha='right',fontsize=15)
            ax.tick_params(labelsize=15)
            plt.tick_params(labelsize=15)
            
    plt.savefig('C:/Users/fanwang/Desktop/tw_hist_ssp245_ssp585.pdf',bbox_inches = 'tight')
    plt.show()

data0=scio.loadmat(r'E:\MG\Research\downscaling paper\CMIP6\scripts\Tw_hist.mat')
tw_hist=np.array(data0['Tw_3']).T

data1=scio.loadmat(r'E:\MG\Research\downscaling paper\CMIP6\scripts\Tw_ssp245.mat')
tw_245=np.array(data1['Tw_245']).T

data2=scio.loadmat(r'E:\MG\Research\downscaling paper\CMIP6\scripts\Tw_ssp585.mat')
tw_585=np.array(data2['Tw_585']).T

tw_585_summer=np.nanmean((tw_585[151:243,:,:]+tw_585[151+365:243+365,:,:]
               +tw_585[151+730:243+730,:,:]+tw_585[151+1095:243+1095,:,:]
               +tw_585[151+1460:243+1460,:,:])/5,0)

tw_hist_summer=np.nanmean((tw_hist[151:243,:,:]+tw_hist[151+365:243+365,:,:]
               +tw_hist[151+730:243+730,:,:]+tw_hist[151+1095:243+1095,:,:]
               +tw_hist[151+1460:243+1460,:,:])/5,0)

tw_245_summer=np.nanmean((tw_245[151:243,:,:]+tw_245[151+365:243+365,:,:]
               +tw_245[151+730:243+730,:,:]+tw_245[151+1095:243+1095,:,:]
               +tw_245[151+1460:243+1460,:,:])/5,0)

ll=xr.open_dataset(r'E:\MG\Research\downscaling paper\CMIP6\scripts\lat_lon.nc')
lat=ll.XLAT.data[0,:,:]
lon=ll.XLONG.data[0,:,:]

plot_china(lat, lon, [tw_hist_summer,tw_245_summer,tw_585_summer], 0, 30)

