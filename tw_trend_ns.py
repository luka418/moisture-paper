# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:25:40 2023

@author: fanwang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import matplotlib.mlab as mlab
from scipy.stats import norm
import os
import glob2
import scipy.stats as st

plt.rcParams['font.sans-serif'] = "Arial"

year_urb0=np.loadtxt('D:/paper_moisture/Data/year_urb.txt')
ll_sel = np.loadtxt(r'D:/paper_moisture/Data/ll_sel2.txt')
mois_p=np.loadtxt('D:/paper_moisture/Data/mois_p_summer.txt')
mois=np.loadtxt('D:/paper_moisture/Data/mois_summer.txt')
dir = r'D:/Data_sets/meteo_each_day/txt_data/'
#sta_f = '/Users/wangfan/Desktop/aws_decode/fz_reg_all.txt'
os.chdir(dir)
files = glob2.glob('*.txt')
out=[]
ll_sel = np.loadtxt(r'D:/paper_moisture/Data/ll_sel2.txt')
stations = ll_sel[:,0]
year_ave=np.loadtxt('D:/paper_moisture/Data/tw_all.txt')
year_ave1=np.zeros((826,40))
for ii in range(826):
    year_ave1[ii,:]=year_ave[ii,19::]-np.nanmean(year_ave[ii,19::])
    
#year_ave1=year_ave[:,19::]

#index_p = np.where((mois<0) & (mois_p<0.05) & (ll_sel[:,4]<33))
index_p = np.where((ll_sel[:,4]<33))
index_p2 = np.where((ll_sel[:,4]>=33))

year_ave_s=year_ave1[index_p[0],:]
year_ave_n=year_ave1[index_p2[0],:]

d_nu = year_ave_s
d_u = year_ave_n

d_nu_a = np.nanmean(d_nu,0)
d_u_a = np.nanmean(d_u,0)

x = np.arange(40)
y_nu_e1 = [np.nanpercentile(d_nu[:,i], 25) for i in range(40)]
y_nu_e2 = [np.nanpercentile(d_nu[:,i], 75) for i in range(40)]

y_u_e1 = [np.nanpercentile(d_u[:,i], 25) for i in range(40)]
y_u_e2 = [np.nanpercentile(d_u[:,i], 75) for i in range(40)]

f0 = np.polyfit(x, d_nu_a, 1)
yvals0 = np.polyval(f0, x)

f1 = np.polyfit(x, d_u_a, 1)
yvals1 = np.polyval(f1, x)

fig = plt.figure(figsize=(9.5, 6), dpi=100)
plt.plot(d_nu_a,'#e6550d',linewidth=2)
plt.plot(yvals0,'--',color='#e6550d',linewidth=1.5)
plt.fill_between(x, y_nu_e1, y_nu_e2, color='#e6550d', alpha=0.2)
plt.text(2,1.7,'0.07 k/decade (p<0.05) South',fontsize=16,color='#e6550d')
plt.plot(d_u_a,'#2c7fb8',linewidth=2)
plt.plot(yvals1,'--',color='#2c7fb8',linewidth=1.5)
plt.fill_between(x, y_u_e1, y_u_e2,color='#2c7fb8', alpha=0.2)
plt.text(2,1.5,'0.23 k/decade (p<0.05) North',fontsize=16,color='#2c7fb8')
plt.xlim(0,39)
plt.ylim(-1.2,2)
plt.ylabel('Tw anomaly ($^\circ$C)',fontsize=19)
plt.xlabel('Year',fontsize=19)
#plt.yticks([-0.08,-0.04,0,0.04,0.08,0.12])
plt.xticks(np.arange(0,40,4),['1979','1983','1987','1991','1995','1999','2003','2007','2011','2015'])
plt.tick_params(labelsize=16)
plt.savefig('C:/Users/fanwang/Desktop/tw_trend.pdf',bbox_inches = 'tight')
