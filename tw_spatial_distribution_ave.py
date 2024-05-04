# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:21:29 2023

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
import math

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
plt.rcParams['font.sans-serif'] ="Arial"

def add_Chinese_provinces(ax, **kwargs):
    proj = ccrs.PlateCarree()
    shp_filepath = 'D:/Python Programes/china_shp/bou2_4p.shp'
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
        #ax.text(135,16,months_names[i],va='bottom', ha='right',fontsize=13)
        #ax.text(136.5,54,'(a)',va='top', ha='right',fontsize=13)
        ax.tick_params(labelsize=12)
        plt.tick_params(labelsize=12)
        
        shp_file = 'D:/Python Programes/china_shp/bou2_4p.shp'
        
        p = ax.scatter(lon, lat, c=diff, s=5, cmap=cmaps.NCV_jet,vmin=vmin, vmax=vmax,marker = "o")
    
        
        ax.plot([103,107],[32,32],'-k',linewidth=1.5)# scb
        ax.plot([103,107],[29,29],'-k',linewidth=1.5)
        ax.plot([103,103],[29,32],'-k',linewidth=1.5)# 40-60, -10-10
        ax.plot([107,107],[29,32],'-k',linewidth=1.5)
    
        ax.plot([112,115],[21.5,21.5],'-k',linewidth=1.5)# 40-60, -10-10
        ax.plot([112,115],[24,24],'-k',linewidth=1.5)
        ax.plot([112,112],[21.5,24],'-k',linewidth=1.5)# 40-60, -10-10
        ax.plot([115,115],[21.5,24],'-k',linewidth=1.5)
        
        ax.plot([118,123],[33,33],'-k',linewidth=1.5)# 40-60, -10-10
        ax.plot([118,123],[29,29],'-k',linewidth=1.5)
        ax.plot([118,118],[29,33],'-k',linewidth=1.5)# 40-60, -10-10
        ax.plot([123,123],[29,33],'-k',linewidth=1.5)
        
        ax.plot([115,120],[38,38],'-k',linewidth=1.5)# 40-60, -10-10
        ax.plot([115,120],[41,41],'-k',linewidth=1.5)
        ax.plot([115,115],[38,41],'-k',linewidth=1.5)# 40-60, -10-10
        ax.plot([120,120],[38,41],'-k',linewidth=1.5)
    axgr.cbar_axes[0].colorbar(p)
    plt.savefig('C:/Users/fanwang/Desktop/fig_moisture_new/tw_summer_ave.eps',bbox_inches = 'tight')
    plt.show()


dir = r'D:/Data_sets/meteo_each_day/txt_data/'
sta_f = '/Users/wangfan/Desktop/aws_decode/fz_reg_all.txt'
os.chdir(dir)
files = glob2.glob('*.txt')
out=[]
ll_sel = np.loadtxt(r'D:/paper_moisture/Data/ll_sel2.txt')
stations = ll_sel[:,0]
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
            year_ave[ss,int(y0)-1960] = np.nanmean(tmp[index2[0],3]*np.arctan(0.151977*(tmp[index2[0],4]+8.313659)**0.5)
                                        +np.arctan(tmp[index2[0],3]+tmp[index2[0],4])
                                        -np.arctan(tmp[index2[0],4]-1.676331)
                                        +0.00391838*(tmp[index2[0],4])**1.5*np.arctan(0.023101*tmp[index2[0],4])-4.686035)


year_ave[year_ave==0]=np.nan
mois = np.nanmean(year_ave[:,19::],1)
diff = mois
add_dots(ll_sel[:,5],ll_sel[:,4],diff,9,29)