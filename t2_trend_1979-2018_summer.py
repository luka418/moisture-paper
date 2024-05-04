# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:35:27 2023

@author: fanwang
"""

import numpy as np
import os
import glob2
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
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
plt.rcParams['font.sans-serif'] ="Arial"

def add_Chinese_provinces(ax, **kwargs):
    proj = ccrs.PlateCarree()
    shp_filepath = 'D:/Python Programes/china_shp/bou2_4p.shp'
    #shp_filepath =(r'C:\Users\fanwang\Desktop\ECo-geo\ECO-geo.shp')
    reader = Reader(shp_filepath)
    provinces = cfeature.ShapelyFeature(reader.geometries(), proj)
    ax.add_feature(provinces, **kwargs)

def add_dots(lon, lat,diff,vmin,vmax):    
    fig = plt.figure(figsize=(10, 4), dpi=100)
    #plt.subplots_adjust(left=0.05, bottom=0.2, right=0.99, top=0.97, hspace=0.2, wspace=0.2)
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(1, 1),
                    axes_pad=0.5,
                    cbar_location='bottom',#'right',
                    cbar_mode='each',
                    cbar_pad=0.3,
                    cbar_size='2%',
                    label_mode='')  # note the empty label_mode
    for i, ax in enumerate(axgr):
        ax.coastlines(resolution='10m', lw=0.3)
        add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
        ax.set_xlim(70, 137)
        ax.set_xticks(np.linspace(70, 130, 4), crs=projection)
        ax.set_ylim(15, 55)
        ax.set_yticks(np.linspace(15, 55, 3), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        ax.tick_params(labelsize=12)
        plt.tick_params(labelsize=12)

        shp_file = 'D:/Python Programes/china_shp/bou2_4p.shp'
        
        p = ax.scatter(lon, lat, c=diff, s=10, cmap=cmaps.BlueWhiteOrangeRed,vmin=vmin, vmax=vmax,marker = "o")
            
    axgr.cbar_axes[0].colorbar(p)
    #ax.set_title(title, fontsize='medium')
    plt.savefig('C:/Users/fanwang/Desktop/spatial_t2_trend_summer_new.eps',bbox_inches = 'tight')
    plt.show()


dir = r'D:/Data_sets/meteo_each_day/txt_data/'
#sta_f = '/Users/wangfan/Desktop/aws_decode/fz_reg_all.txt'
os.chdir(dir)
files = glob2.glob('*.txt')
out=[]
ll_sel = np.loadtxt(r'D:/paper_moisture/Data/ll_sel2.txt')
stations = ll_sel[:,0]
mark = np.loadtxt(r'D:/paper_moisture/Data/yz.txt')

year_ave = np.zeros((826,59))
year_ave_rh = np.zeros((826,59))
year_ave_t = np.zeros((826,59))
for ff in files:
    print(ff)
    data1 = np.loadtxt(r'D:/Data_sets/meteo_each_day/txt_data/'+ff)
    index=np.where(np.isin(data1[:,0], stations))
    dd=np.column_stack((data1[index[0],0:3],data1[index[0],4:6]))
    for ss in range(826):
        index1 = np.where(dd[:,0]==stations[ss])
        tmp = dd[index1[0],:]
        for y0 in (np.unique(tmp[:,1])):
            index2 = np.where((tmp[:,1]==y0) & ((tmp[:,2]>5) & (tmp[:,2]<9))) #winter
            year_ave[ss,int(y0)-1960] = np.nanmean(6.112*np.exp(17.62*(tmp[index2[0],3])/(tmp[index2[0],3]+243.12))*tmp[index2[0],4]/100)
            year_ave_t[ss,int(y0)-1960] = np.nanmean(tmp[index2[0],3])
            year_ave_rh[ss,int(y0)-1960] = np.nanmean(tmp[index2[0],4])
year_ave[year_ave==0]=np.nan
year_ave_t[year_ave_t==0]=np.nan
year_ave_rh[year_ave_rh==0]=np.nan
mois = np.zeros((826))
mois_p = np.zeros((826))
rh = np.zeros((826))
t = np.zeros((826))
for i in range(826):
    mois[i] = stats.linregress(np.arange(40), year_ave_t[i,19::])[0]
    mois_p[i] = stats.linregress(np.arange(40), year_ave_t[i,19::])[3]
    
index_p = np.where((mois_p<0.05))
diff = mois
add_dots(ll_sel[index_p[0],5],ll_sel[index_p[0],4],diff[index_p[0]]*10,-0.6,0.6)
