# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:25:43 2023

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
year_ave=np.loadtxt('D:/paper_moisture/Data/tw_all.txt')
ll_sel = np.loadtxt(r'D:/paper_moisture/Data/ll_sel2.txt')
year_ave1=year_ave[:,19::]

index_p1 = np.where((ll_sel[:,4]<=33) & (ll_sel[:,4]>=29) & (ll_sel[:,5]<=123) & (ll_sel[:,5]>=118))
index_p2 = np.where((ll_sel[:,4]<=23) & (ll_sel[:,4]>=21) & (ll_sel[:,5]<=115) & (ll_sel[:,5]>=112))
index_p3 = np.where((ll_sel[:,4]<=41) & (ll_sel[:,4]>=38) & (ll_sel[:,5]<=120) & (ll_sel[:,5]>=115))
index_p4 = np.where((ll_sel[:,4]<=32) & (ll_sel[:,4]>=29) & (ll_sel[:,5]<=107) & (ll_sel[:,5]>=103))

year_ave_yrd=year_ave1[index_p1[0],:]
year_ave_prd=year_ave1[index_p2[0],:]
year_ave_bth=year_ave1[index_p3[0],:]
year_ave_scb=year_ave1[index_p4[0],:]

yrd_a = np.nanmean(year_ave_yrd,0)
prd_a = np.nanmean(year_ave_prd,0)
bth_a = np.nanmean(year_ave_bth,0)
scb_a = np.nanmean(year_ave_scb,0)

x = np.arange(40)
y_yrd_e1 = [np.nanpercentile(year_ave_yrd[:,i], 25) for i in range(40)]
y_yrd_e2 = [np.nanpercentile(year_ave_yrd[:,i], 75) for i in range(40)]

y_prd_e1 = [np.nanpercentile(year_ave_prd[:,i], 25) for i in range(40)]
y_prd_e2 = [np.nanpercentile(year_ave_prd[:,i], 75) for i in range(40)]

y_bth_e1 = [np.nanpercentile(year_ave_bth[:,i], 25) for i in range(40)]
y_bth_e2 = [np.nanpercentile(year_ave_bth[:,i], 75) for i in range(40)]

y_scb_e1 = [np.nanpercentile(year_ave_scb[:,i], 25) for i in range(40)]
y_scb_e2 = [np.nanpercentile(year_ave_scb[:,i], 75) for i in range(40)]

f0 = np.polyfit(x, yrd_a, 1)
yvals0 = np.polyval(f0, x)

f1 = np.polyfit(x, prd_a, 1)
yvals1 = np.polyval(f1, x)

f2 = np.polyfit(x, bth_a, 1)
yvals2 = np.polyval(f2, x)

f3 = np.polyfit(x, scb_a, 1)
yvals3 = np.polyval(f3, x)
#st.linregress(x,bth_a)

fig = plt.figure(figsize=(9.5, 6), dpi=100)
plt.plot(yrd_a,'#e6550d',linewidth=2)
plt.plot(yvals0,'--',color='#e6550d',linewidth=1.5)
plt.fill_between(x, y_yrd_e1, y_yrd_e2, color='#e6550d', alpha=0.2)
plt.text(8,27,'0.11 k/decade YRD',fontsize=16,color='#e6550d')

plt.plot(prd_a,'#2c7fb8',linewidth=2)
plt.plot(yvals1,'--',color='#2c7fb8',linewidth=1.5)
plt.fill_between(x, y_prd_e1, y_prd_e2,color='#2c7fb8', alpha=0.2)
plt.text(22,27,'0.08 k/decade PRD',fontsize=16,color='#2c7fb8')

plt.plot(bth_a,'#31a354',linewidth=2)
plt.plot(yvals2,'--',color='#31a354',linewidth=1.5)
plt.fill_between(x, y_bth_e1, y_bth_e2, color='#31a354', alpha=0.2)
plt.text(8,27.5,'0.16 k/decade BTH',fontsize=16,color='#31a354')

plt.plot(scb_a,'#8856a7',linewidth=2)
plt.plot(yvals3,'--',color='#8856a7',linewidth=1.5)
plt.fill_between(x, y_scb_e1, y_scb_e2,color='#8856a7', alpha=0.2)
plt.text(22,27.5,'0.02 k/decade SCB',fontsize=16,color='#8856a7')

plt.xlim(0,39)
plt.ylim(19,28)
plt.ylabel('Tw ($^\circ$C)',fontsize=19)
plt.xlabel('Year',fontsize=19)
#plt.yticks([-0.08,-0.04,0,0.04,0.08,0.12])
plt.xticks(np.arange(0,40,4),['1979','1983','1987','1991','1995','1999','2003','2007','2011','2015'])
plt.tick_params(labelsize=16)
#plt.savefig('C:/Users/fanwang/Desktop/tw_trend_rg.pdf',bbox_inches = 'tight')
