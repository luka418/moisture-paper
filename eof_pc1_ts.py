# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:27:31 2024

@author: fanwang
"""


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
from mpl_toolkits.axes_grid1 import AxesGrid
import cmaps
import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from eofs.standard import Eof

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
    
def plot_china(lat1, lon1, data, data2, vmin, vmax):  
    fig = plt.figure(figsize=(10, 4), dpi=100)
    #plt.subplots_adjust(left=0.05, bottom=0.2, right=0.99, top=0.97, hspace=0.2, wspace=0.2)
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(1, 1),
                    axes_pad=0.5,
                    cbar_location='bottom',
                    cbar_mode='each',#'single',
                    cbar_pad=0.3,
                    cbar_size='2%',
                    label_mode='')  # note the empty label_mode
    for i, ax in enumerate(axgr):
        # 添加海岸线和中国省界.
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
                levels=np.linspace(vmin[i], vmax[i], 21), 
                cmap=cmaps.temp_19lev,
                extend='both'
                )

            axgr.cbar_axes[i].colorbar(p)
            add_Chinese_provinces(ax, lw=0.5, ec='k', fc='none')
            ax.coastlines(resolution='10m', lw=0.3)
            
            p2 = ax.contourf(lon1,lat1, data2[i],levels=[np.nanmin(data2[i]),0.05,np.nanmax(data2[i])],zorder=1,hatches=['...', None],colors="none",extend='none')
            
            for ii, collection in enumerate(p2.collections):
                collection.set_edgecolor('k')
            for collection in p2.collections:
                collection.set_linewidth(0.)

            ax.set_xlim(60, 180)
            ax.set_xticks([60,100,140,180], crs=projection)
            ax.set_ylim(5, 65)
            ax.set_yticks([5,25,45,65], crs=projection)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            #ax.text(180,90,"%.2f" %names[i], va='bottom', ha='right',fontsize=15)
            ax.tick_params(labelsize=12)
            plt.tick_params(labelsize=12)
            
            ax.plot([80,120],[33,33],'b',linewidth=2)# 40-60, -10-10
            ax.plot([80,120],[50,50],'b',linewidth=2)
            ax.plot([80,80],[33,50],'b',linewidth=2)
            ax.plot([120,120],[33,50],'b',linewidth=2)
    #plt.savefig('C:/Users/fanwang/Desktop/t_regression_pc1.eps',bbox_inches = 'tight')
    plt.show()
    #plt.close()
    
year_ave=np.loadtxt('D:/paper_moisture/Data/tw_all.txt')
ll_sel = np.loadtxt(r'D:/paper_moisture/Data/ll_sel2.txt')

nan_num = []
for i in range(826):
    nan_num.append(np.sum(np.isnan(year_ave[i,:])))
index = np.where(np.array(nan_num)==0)

year_ave=year_ave[index[0],:]
ll_sel=ll_sel[index[0],:]
#coslat = np.cos(np.deg2rad(lat1))
#wgts = np.sqrt(coslat)#[..., np.newaxis]
solver = Eof(year_ave.T)
eof = solver.eofsAsCorrelation(neofs=10)
pc = solver.pcs(npcs=10)
var = solver.varianceFraction()

data=xr.open_dataset(r'D:/paper_moisture/ec_file/skt_1979-2018.nc')

skt=data.skt.data
lon1, lat1 = np.array(data.longitude),np.array(data.latitude)
lon, lat = np.meshgrid(lon1[240:721], lat1[100:341])
#lon, lat = lon1, lat1

td = np.reshape(skt,(40,3,721,1440))
e_a = np.nanmean(td,1)[:,100:341,240:721]

trend_e = np.zeros((241,481))
p_e = np.zeros((241,481))
for i in range(241):
    for j in range(481):
        trend_e[i,j] = stats.linregress(pc[:,0], e_a[:,i,j])[0]
        p_e[i,j] = stats.linregress(pc[:,0], e_a[:,i,j])[3]

plot_china(lat, lon, [trend_e*10],[p_e],[-1], [1])

x=np.arange(1979,2018)
f0 = np.polyfit(x,pc[:,0], 1)
yvals0 = np.polyval(f0, x)

f1 = np.polyfit(x, t_ave-np.mean(t_ave), 1)
yvals1 = np.polyval(f1, x)

fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=100)
ax1.plot(np.arange(1979,2018),pc[:,0],'#e6550d',linewidth=3)
ax1.plot(x,yvals0,'--',color='#e6550d',linewidth=2)
ax2 = ax1.twinx()
ax2.plot(np.arange(1979,2018),t_ave-np.mean(t_ave),'#2c7fb8',linewidth=3)
ax2.plot(x,yvals1,'--',color='#2c7fb8',linewidth=2)
ax1.text(2000,-14,'Ts: y=0.51x-0.97 (p<0.01) ',fontsize=16,color='#2c7fb8')
ax1.text(2000,-16.5,'PC1: y=0.50x-9.32 (p<0.01) ',fontsize=16,color='#e6550d')
ax1.set_ylabel('PC1',fontsize=19)
ax2.set_ylabel('T$_s$ anomaly ($^{\circ}$C)',fontsize=19)
ax1.set_xlabel('Year',fontsize=19)
ax1.set_xlim(1979,2017)
#plt.yticks([-0.08,-0.04,0,0.04,0.08,0.12])
ax1.set_xticks(np.arange(1979,2018,5))
ax1.tick_params(labelsize=15)
ax2.tick_params(labelsize=15)
plt.savefig('C:/Users/fanwang/Desktop/ts_pc1.eps',bbox_inches = 'tight')

stats.linregress(np.arange(39),t_ave-np.mean(t_ave))

fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.plot(t_ave-np.mean(t_ave),pc[:,0],'o',color='#c51b8a')
ax.plot([-2,2],[-20,20],'--k', linewidth=2)
ax.set_ylim(-20,20)
ax.set_xlim(-2,2)
ax.set_aspect(0.1)
ax.set_ylabel('PC1',fontsize=17)
ax.set_xlabel('T$_s$ anomaly ($^{\circ}$C)',fontsize=17)
ax.tick_params(labelsize=15)
plt.savefig('C:/Users/fanwang/Desktop/ts_pc1_cor.eps',bbox_inches = 'tight')
