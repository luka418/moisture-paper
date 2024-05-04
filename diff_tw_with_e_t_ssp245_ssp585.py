# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:57:53 2023

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
    fig = plt.figure(figsize=(10, 8), dpi=100)
    #plt.subplots_adjust(left=0.05, bottom=0.2, right=0.99, top=0.97, hspace=0.2, wspace=0.2)
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(1, 2),
                    axes_pad=0.5,
                    label_mode='',
                    cbar_location='right',
                    cbar_mode='single',#'single',
                    #share_all=True,
                    cbar_pad=0.1,
                    cbar_size='4%',
                    )  # note the empty label_mode
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
                levels=np.linspace(vmin, vmax, 21), 
                cmap=cmaps.WhiteBlueGreenYellowRed[50::],
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
            ax.tick_params(labelsize=12)
            plt.tick_params(labelsize=12)
            
    plt.savefig('C:/Users/fanwang/Desktop/diff_tw_t_e_ssp245.pdf',bbox_inches = 'tight')
    plt.show()

data0=scio.loadmat(r'E:\MG\Research\downscaling paper\CMIP6\scripts\TRH_hist.mat')
t_hist=np.array(data0['T_2010']).T
rh_hist=np.array(data0['RH_2010']).T
e_hist=6.112*np.exp(17.62*t_hist/(t_hist+243.12))*rh_hist/100

data1=scio.loadmat(r'E:\MG\Research\downscaling paper\CMIP6\scripts\TRH_ssp245.mat')
t_245=np.array(data1['T_ssp245']).T
rh_245=np.array(data1['RH_ssp245']).T
e_245=6.112*np.exp(17.62*t_245/(t_245+243.12))*rh_245/100

data2=scio.loadmat(r'E:\MG\Research\downscaling paper\CMIP6\scripts\TRH_ssp585.mat')
t_585=np.array(data2['T_ssp585']).T
rh_585=np.array(data2['RH_ssp585']).T
e_585=6.112*np.exp(17.62*t_585/(t_585+243.12))*rh_585/100


def tw_cal(t2,e):
    rh=e/(6.112*np.exp(17.62*t2/(t2+243.12)))*100
    tw=t2*np.arctan(0.151977*(rh+8.313659)**0.5)+np.arctan(t2+rh)-np.arctan(rh-1.676331)+0.00391838*(rh)**1.5*np.arctan(0.023101*rh)-4.686035
    return tw

tw_245_t=tw_cal(t_hist,e_245)
tw_245_e=tw_cal(t_245,e_hist)
tw_585_t=tw_cal(t_hist[0:-1,:,:],e_585)
tw_585_e=tw_cal(t_585,e_hist[0:-1,:,:])

data0=scio.loadmat(r'E:\MG\Research\downscaling paper\CMIP6\scripts\Tw_hist.mat')
tw_hist=np.array(data0['Tw_3']).T

tw_hist_summer=np.nanmean((tw_hist[151:243,:,:]+tw_hist[151+365:243+365,:,:]
               +tw_hist[151+730:243+730,:,:]+tw_hist[151+1095:243+1095,:,:]
               +tw_hist[151+1460:243+1460,:,:])/5,0)

tw_245_t_summer=np.nanmean((tw_245_t[151:243,:,:]+tw_245_t[151+365:243+365,:,:]
               +tw_245_t[151+730:243+730,:,:]+tw_245_t[151+1095:243+1095,:,:]
               +tw_245_t[151+1460:243+1460,:,:])/5,0)

tw_245_e_summer=np.nanmean((tw_245_e[151:243,:,:]+tw_245_e[151+365:243+365,:,:]
               +tw_245_e[151+730:243+730,:,:]+tw_245_e[151+1095:243+1095,:,:]
               +tw_245_e[151+1460:243+1460,:,:])/5,0)

tw_585_t_summer=np.nanmean((tw_585_t[151:243,:,:]+tw_585_t[151+365:243+365,:,:]
               +tw_585_t[151+730:243+730,:,:]+tw_585_t[151+1095:243+1095,:,:]
               +tw_585_t[151+1460:243+1460,:,:])/5,0)

tw_585_e_summer=np.nanmean((tw_585_e[151:243,:,:]+tw_585_e[151+365:243+365,:,:]
               +tw_585_e[151+730:243+730,:,:]+tw_585_e[151+1095:243+1095,:,:]
               +tw_585_e[151+1460:243+1460,:,:])/5,0)

ll=xr.open_dataset(r'E:\MG\Research\downscaling paper\CMIP6\scripts\lat_lon.nc')
lat=ll.XLAT.data[0,:,:]
lon=ll.XLONG.data[0,:,:]

plot_china(lat, lon, [tw_245_e_summer-tw_hist_summer,tw_245_t_summer-tw_hist_summer],0,5)
