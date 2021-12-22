# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:34:04 2020

@author: mkuban
"""

import numpy as np
import dclab
import matplotlib.pylab as plt
from matplotlib import cm
from scipy.stats import gaussian_kde

cmap_vir = cm.get_cmap('viridis')


def density_scatter( x , y, bins, ax, sort = True, **kwargs )   :

    np.nan_to_num(y, copy=False)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    ax.scatter( x, y, c=z, cmap = cmap_vir, marker = ".", s = 20, picker = True, **kwargs )    
    plt.subplots_adjust(wspace = 0.5)
    ax.tick_params(direction ='in')
    plt.rcParams["font.size"] = 20
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    plt.gcf().set_tight_layout(False)
    return ax




#%%
############################################### FILE LOADING #######################################
filepath = r"Q:\Data\M001" # change filepath
ds = dclab.new_dataset(filepath + ".rtdc")


#%%
############################################## FILTERS ##########################################

ds.config["filtering"]["area_um min"] = 25
ds.config["filtering"]["area_um max"] = 600
ds.config["filtering"]["aspect min"] = 1
ds.config["filtering"]["aspect max"] = 2
ds.config["filtering"]["area_ratio min"] = 1
ds.config["filtering"]["area_ratio max"] = 1.05

ds.apply_filter() 
val1 = ds.filter.all # valid events

ds = dclab.new_dataset(filepath + ".rtdc") ################### CD45+ others-
ds.config["filtering"]["area_um min"] = 25
ds.config["filtering"]["area_um max"] = 600
ds.config["filtering"]["aspect min"] = 1
ds.config["filtering"]["aspect max"] = 2
ds.config["filtering"]["area_ratio min"] = 1
ds.config["filtering"]["area_ratio max"] = 1.05
ds.config["filtering"]["fl1_max min"] = -100
ds.config["filtering"]["fl1_max max"] = 200
ds.config["filtering"]["fl2_max min"] = 300
ds.config["filtering"]["fl2_max max"] = 200000
ds.config["filtering"]["fl3_max min"] = -100
ds.config["filtering"]["fl3_max max"] = 200

ds.apply_filter() 
val2 = ds.filter.all # valid events


ds = dclab.new_dataset(filepath + ".rtdc") ################### EPCAM+ others-
ds.config["filtering"]["area_um min"] = 25
ds.config["filtering"]["area_um max"] = 600
ds.config["filtering"]["aspect min"] = 1
ds.config["filtering"]["aspect max"] = 2
ds.config["filtering"]["area_ratio min"] = 1
ds.config["filtering"]["area_ratio max"] = 1.08
ds.config["filtering"]["fl1_max min"] = -100
ds.config["filtering"]["fl1_max max"] = 200
ds.config["filtering"]["fl2_max min"] = -100
ds.config["filtering"]["fl2_max max"] = 200
ds.config["filtering"]["fl3_max min"] = 300
ds.config["filtering"]["fl3_max max"] = 200000

ds.apply_filter() 
val3 = ds.filter.all # valid events

ds = dclab.new_dataset(filepath + ".rtdc")



#%%

figure = plt.figure(figsize=(20,10))
ax = plt.subplot(241, xlabel = 'Cell size [$\mu$m$^2$]', xlim = (10,200), ylabel = 'Deformation [a.u.]', ylim = (0, 0.15))
density_scatter(ds["area_um"], ds["deform"], bins = [1000,100], ax = ax)
ax = plt.subplot(242, xlabel = 'Cell size [$\mu$m$^2$]', xlim = (10,150), ylabel = 'Deformation [a.u.]', ylim = (0, 0.2))
density_scatter(ds["area_um"][val1], ds["deform"][val1], bins = [1000,100], ax = ax)
ax = plt.subplot(243, xlabel = 'Cell size [$\mu$m$^2$]', xlim = (10,150), ylabel = 'Deformation [a.u.]', ylim = (0, 0.2))
density_scatter(ds["area_um"][val2], ds["deform"][val2], bins = [1000,100], ax = ax)
ax = plt.subplot(244, xlabel = 'Cell size [$\mu$m$^2$]', xlim = (10,150), ylabel = 'Deformation [a.u.]', ylim = (0, 0.2))
density_scatter(ds["area_um"][val3], ds["deform"][val3], bins = [1000,100], ax = ax)
ax = plt.subplot(245, xlabel = 'Cell size [$\mu$m$^2$]', xlim = (10,150), ylabel = 'Brightness [a.u.]', ylim = (70, 160))
density_scatter(ds["area_um"], ds["bright_avg"], bins = [1000,100], ax = ax)
ax = plt.subplot(246, xlabel = 'Cell size [$\mu$m$^2$]', xlim = (10,150), ylabel = 'Brightness [a.u.]', ylim = (70, 160))
density_scatter(ds["area_um"][val1], ds["bright_avg"][val1], bins = [1000,100], ax = ax)
ax = plt.subplot(247, xlabel = 'Cell size [$\mu$m$^2$]', xlim = (10,200), ylabel = 'Brightness [a.u.]', ylim = (70, 160))
density_scatter(ds["area_um"][val2], ds["bright_avg"][val2], bins = [1000,100], ax = ax)
ax = plt.subplot(248, xlabel = 'Cell size [$\mu$m$^2$]', xlim = (10,200), ylabel = 'Brightness [a.u.]', ylim = (70, 160))
density_scatter(ds["area_um"][val3], ds["bright_avg"][val3], bins = [1000,100], ax = ax)
plt.show()

