# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:34:04 2020

@author: mkuban
"""

import numpy as np
import dclab
import matplotlib.pylab as plt
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde, wilcoxon, norm
import seaborn as sns
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cmap_vir = cm.get_cmap('viridis')

left, width = 0.2, 0.5
bottom, height = 0.1, 0.5
spacing = 0.02      
rect_scatter = [left, bottom, width, height]

def density_scatter(x , y, bins, ax, sort = True, **kwargs )   :

    np.nan_to_num(y, copy=False)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    left, width = 0.2, 0.5
    bottom, height = 0.1, 0.5
    rect_scatter = [left, bottom, width, height]
    plt.figure(figsize=(8, 8))
    ax = plt.axes(rect_scatter)
    ax.tick_params(direction='in', labelleft = True, labelbottom = True)
    
    plt.rcParams["font.size"] = 20
    ax.scatter( x, y, c=z, cmap = cmap_vir, marker = ".", s = 20, picker = True, **kwargs )    
    ax.set_xlim((10, 400))
    ax.set_ylim((0, 0.15))
    ax.set_xlabel('Cell size [$\mu$m$^2$]')
    ax.set_ylabel('Deformation')
    ax.tick_params(direction ='in')
    ax.tick_params(direction ='in', width = 1, length = 4, which = 'both', grid_alpha = 0.2)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    plt.gcf().set_tight_layout(False)

    return ax


def density_scatter2( x , y, x2, y2, bins, ax, sort = True, **kwargs )   :

    np.nan_to_num(y)
    xy = np.vstack([x2,y2])
    z = gaussian_kde(xy)(xy)

    left, width = 0.2, 0.5
    bottom, height = 0.1, 0.5
    spacing = 0.02
        
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.1]
    rect_histy = [left + width + spacing, bottom, 0.1, height]

    plt.figure(figsize=(8, 8))  
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.set_xlim((40, 300))
    ax_scatter.set_ylim((0, 0.15))
    sns.kdeplot(x2, y2, levels = [0.5, 0.95, 1], fill = True, shade_lowest = False, cmap='Greens', alpha = 0.8, label = 'Tumour', ax = ax_scatter)
    sns.kdeplot(x, y, levels = [0.5, 0.95, 1], fill = True, shade_lowest = False, cmap = "Purples", alpha = 0.6, label = 'Healthy tissue', ax = ax_scatter)
    ax_scatter.set_xlabel('Cell size [$\mu$m$^2$]')
    ax_scatter.set_ylabel('Deformation')
    ax_scatter.tick_params(direction ='in')
    
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', which = 'both', labelleft=False,  labelbottom=False)
    sns.distplot(x, bins = 100, hist = False, color = '#ab80d5', norm_hist =False, kde= True, kde_kws={"shade": True, "alpha": 0.6}, hist_kws=dict(alpha=0.1), ax = ax_histx)
    sns.distplot(x2, bins = 40, hist = False, color = '#1e8549', norm_hist =False, kde= True, kde_kws={"shade": True, "alpha": 0.4}, hist_kws=dict(alpha=0.9), ax = ax_histx)
    ax_histx.set_xlabel(' ')     
    ax_histx.set_ylabel(' ')     
    ax_histx.set_xlim(ax_scatter.get_xlim())
    
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', which = 'both', labelleft=False,  labelbottom=False)
    sns.distplot(y, hist = False, color = '#ab80d5', norm_hist =False, kde= True, kde_kws={"shade": True, "alpha": 0.6}, hist_kws=dict(alpha=0.1), ax = ax_histy, vertical = True)
    sns.distplot(y2, hist = False, color = '#1e8549', norm_hist =False, kde= True, kde_kws={"shade": True, "alpha": 0.4}, hist_kws=dict(alpha=0.9), ax = ax_histy, vertical = True)
    ax_histy.set_ylabel(' ')      
    ax_histy.set_ylim(ax_scatter.get_ylim())
                     
    plt.rcParams["font.size"] = 18
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.gcf().set_tight_layout(False)
    
    return ax_scatter



def density_scatter3( x , y, x2, y2, bins, ax, sort = True, **kwargs )   :

    np.nan_to_num(y)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # definitions for the axes
    left, width = 0.2, 0.5
    bottom, height = 0.1, 0.5
    spacing = 0.02
        
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.1]
    rect_histy = [left + width + spacing, bottom, 0.1, height]

    plt.figure(figsize=(8, 8))
    
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', labelleft = True, labelbottom = True)
    ax_scatter.scatter( x, y, c=z, cmap = cmap_vir, marker = ".", s = 20, picker = True, **kwargs) 
    ax_scatter.set_xlim((10, 200))
    ax_scatter.set_ylim((0, 0.15))
    ax_scatter.set_xlabel('Cell size [$\mu$m$^2$]')
    ax_scatter.set_ylabel('Deformation [a.u.]')
    ax_scatter.tick_params(direction ='in')
   
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', which = 'both', labelleft=False,  labelbottom=False)
    sns.distplot(x, bins = 100, hist = False, color = '#ab80d5', norm_hist =False, kde= True, kde_kws={"shade": True, "alpha": 0.8}, hist_kws=dict(alpha=0.1), ax = ax_histx)
    sns.distplot(x2, bins = 40, hist = False, color = '#1e8549', norm_hist =False, kde= True, kde_kws={"shade": True, "alpha": 0.1, "linestyle":"--"}, hist_kws=dict(alpha=0.9), ax = ax_histx)
    ax_histx.set_xlabel(' ')     
    ax_histx.set_ylabel(' ')    
    ax_histx.set_xlim(ax_scatter.get_xlim())
    
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', which = 'both', labelleft=False,  labelbottom=False)
    sns.distplot(y, hist = False, color = '#ab80d5', norm_hist =False, kde= True, kde_kws={"shade": True, "alpha": 0.8}, hist_kws=dict(alpha=0.1), ax = ax_histy, vertical = True)
    sns.distplot(y2, hist = False, color = '#1e8549', norm_hist =False, kde= True, kde_kws={"shade": True, "alpha": 0.1, "linestyle":"--"}, hist_kws=dict(alpha=0.9), ax = ax_histy, vertical = True)
    ax_histy.set_ylabel(' ')       
    ax_histy.set_ylim(ax_scatter.get_ylim())
    
    plt.rcParams["font.size"] = 18
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.gcf().set_tight_layout(False)
    
    return ax


def plot_box_swarm(data, parameter):

    plt.figure(figsize = (6,6))
    ax = plt.axes()
    y_pos = np.arange(0.5 , len(data), step = 2) # 4
      
    ax = sns.boxplot(data = data, palette = ["#49246d", "#1e8549"])
    sns.swarmplot(data = data, color = "0.25", size = 6)
    
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))

    plt.xticks(y_pos, labels = ['Healthy', 'Tumour'], rotation = 0)
    plt.ylabel(parameter)
    plt.tick_params(direction ='in')
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4, top = 0.9, bottom = 0.2, left = 0.2, right = 0.8)

    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.5)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    plt.tight_layout()
    plt.show()



plt.rcParams['svg.fonttype'] = 'none'


#%% ADJUST USED PARAMETERS

parameters = [
                          "deformME1",
                          "deformMD1",
                          "deformSTD1",
                          "area_umME1",
                          "area_umMD1",
                          "area_umSTD1",
                          "aspect_ratioME1",
                          "aspect_ratioMD1",
                          "aspect_ratioSTD1",
#                          "bright_avgME1",
#                          "bright_avgMD1",
#                          "bright_avgSTD1",
#                          "bright_sdME1",
#                          "bright_sdMD1",
#                          "bright_sdSTD1",                        
                          "area_ratioME1",
                          "area_ratioMD1",
                          "area_ratioSTD1",
                          "deformME2",
                          "deformMD2",
                          "deformSTD2",
                          "area_umME2",
                          "area_umMD2",
                          "area_umSTD2",
                          "aspect_ratioME2",
                          "aspect_ratioMD2",
                          "aspect_ratioSTD2",
#                          "bright_avgME2",
#                          "bright_avgMD2",
#                          "bright_avgSTD2",
#                          "bright_sdME2",
#                          "bright_sdMD2",
#                          "bright_sdSTD2",                        
                          "area_ratioME2",
                          "area_ratioMD2",
                          "area_ratioSTD2",
                          "deformME3",
                          "deformMD3",
                          "deformSTD3",
                          "area_umME3",
                          "area_umMD3",
                          "area_umSTD3",
                          "aspect_ratioME3",
                          "aspect_ratioMD3",
                          "aspect_ratioSTD3",
#                          "bright_avgME3",
#                          "bright_avgMD3",
#                          "bright_avgSTD3",
#                          "bright_sdME3",
#                          "bright_sdMD3",
#                          "bright_sdSTD3",                        
                          "area_ratioME3",
                          "area_ratioMD3",
                          "area_ratioSTD3",
                          "deformME4",
                          "deformMD4",
                          "deformSTD4",
                          "area_umME4",
                          "area_umMD4",
                          "area_umSTD4",
                          "aspect_ratioME4",
                          "aspect_ratioMD4",
                          "aspect_ratioSTD4",
#                          "bright_avgME4",
#                          "bright_avgMD4",
#                          "bright_avgSTD4",
#                          "bright_sdME4",
#                          "bright_sdMD4",
#                          "bright_sdSTD4",                        
                          "area_ratioME4",
                          "area_ratioMD4",
                          "area_ratioSTD4",
                          "ratio1",
                          "ratio2",
                          "ratio3",
                          "ratio4",
                          "ratio5",
                          "ratio6",
                          "ratio7",
                          "ratio8",
                          "ratio9",
                          "ratio10"]


#%%
############################################### FILE LOADING #######################################

filepaths = (
# controls
    r"Q:\Data\M001,"
    # add  filepaths

# tumours
    r"Q:\Data\M002,"
    # add filepaths
    )


# blind samples for testing
filepaths_blind = (
    r"Q:\Data\RTDC\M001,"
    # add filepaths
    )

    
all_parameters = []

for file in filepaths:
    
##first area gate ##### EPCAM POSITIVE
    ds = dclab.new_dataset(file + ".rtdc")
    print(ds.config["setup"]["chip region"])

    ds.config["filtering"]["area_um min"] = 60
    ds.config["filtering"]["area_um max"] = 90
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["bright_avg min"] = 0
    ds.config["filtering"]["bright_avg max"] = 200
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val1 = ds.filter.all # valid events

## second area gate
    ds = dclab.new_dataset(file + ".rtdc")
    
    ds.config["filtering"]["area_um min"] = 60
    ds.config["filtering"]["area_um max"] = 90
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["bright_avg min"] = 0
    ds.config["filtering"]["bright_avg max"] = 200
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000

    ds.apply_filter() 
    val2 = ds.filter.all # valid events   
    
## third area gate
    ds = dclab.new_dataset(file + ".rtdc")
    
    ds.config["filtering"]["area_um min"] = 80
    ds.config["filtering"]["area_um max"] = 120
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["bright_avg min"] = 0
    ds.config["filtering"]["bright_avg max"] = 200
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val3 = ds.filter.all # valid events


## fourth area gate
    ds = dclab.new_dataset(file + ".rtdc")
    
    ds.config["filtering"]["area_um min"] = 120
    ds.config["filtering"]["area_um max"] = 400
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["bright_avg min"] = 0
    ds.config["filtering"]["bright_avg max"] = 200
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val4 = ds.filter.all # valid events
    
    

    ratio1 = 100*(sum(val1)/(sum(val1)+sum(val2)+sum(val3)+sum(val4)))
    ratio2 = 100*(sum(val2)/(sum(val1)+sum(val2)+sum(val3)+sum(val4)))
    ratio3 = 100*(sum(val3)/(sum(val1)+sum(val2)+sum(val3)+sum(val4)))
    ratio4 = 100*(sum(val4)/(sum(val1)+sum(val2)+sum(val3)+sum(val4)))
    ratio5 = 100*(sum(val1)/sum(val2))
    ratio6 = 100*(sum(val2)/sum(val3))
    ratio7 = 100*(sum(val3)/sum(val1))
    ratio8 = 100*(sum(val4)/sum(val1))
    ratio9 = 100*(sum(val4)/sum(val2))
    ratio10 = 100*(sum(val4)/sum(val3))
    
    all_parameters.append([
                          np.mean(ds["deform"][val1]),
                          np.median(ds["deform"][val1]),
                          np.std(ds["deform"][val1]),
                          np.mean(ds["area_um"][val1]),
                          np.median(ds["area_um"][val1]),
                          np.std(ds["area_um"][val1]),
                          np.mean(ds["aspect"][val1]),
                          np.median(ds["aspect"][val1]),
                          np.std(ds["aspect"][val1]),
                          np.mean(ds["bright_avg"][val1]),
                          np.median(ds["bright_avg"][val1]),
                          np.std(ds["bright_avg"][val1]),
                          np.mean(ds["bright_sd"][val1]),
                          np.median(ds["bright_sd"][val1]),
                          np.std(ds["bright_sd"][val1]),                          
                          np.mean(ds["area_ratio"][val1]),
                          np.median(ds["area_ratio"][val1]),
                          np.std(ds["area_ratio"][val1]),
                          np.mean(ds["deform"][val2]),
                          np.median(ds["deform"][val2]),
                          np.std(ds["deform"][val2]),
                          np.mean(ds["area_um"][val2]),
                          np.median(ds["area_um"][val2]),
                          np.std(ds["area_um"][val2]),
                          np.mean(ds["aspect"][val2]),
                          np.median(ds["aspect"][val2]),
                          np.std(ds["aspect"][val2]),
                          np.mean(ds["bright_avg"][val2]),
                          np.median(ds["bright_avg"][val2]),
                          np.std(ds["bright_avg"][val2]),
                          np.mean(ds["bright_sd"][val2]),
                          np.median(ds["bright_sd"][val2]),
                          np.std(ds["bright_sd"][val2]),                          
                          np.mean(ds["area_ratio"][val2]),
                          np.median(ds["area_ratio"][val2]),
                          np.std(ds["area_ratio"][val2]),
                          np.mean(ds["deform"][val3]),
                          np.median(ds["deform"][val3]),
                          np.std(ds["deform"][val3]),
                          np.mean(ds["area_um"][val3]),
                          np.median(ds["area_um"][val3]),
                          np.std(ds["area_um"][val3]),
                          np.mean(ds["aspect"][val3]),
                          np.median(ds["aspect"][val3]),
                          np.std(ds["aspect"][val3]),
                          np.mean(ds["bright_avg"][val3]),
                          np.median(ds["bright_avg"][val3]),
                          np.std(ds["bright_avg"][val3]),
                          np.mean(ds["bright_sd"][val3]),
                          np.median(ds["bright_sd"][val3]),
                          np.std(ds["bright_sd"][val3]),                          
                          np.mean(ds["area_ratio"][val3]),
                          np.median(ds["area_ratio"][val3]),
                          np.std(ds["area_ratio"][val3]),
                          np.mean(ds["deform"][val4]),
                          np.median(ds["deform"][val4]),
                          np.std(ds["deform"][val4]),
                          np.mean(ds["area_um"][val4]),
                          np.median(ds["area_um"][val4]),
                          np.std(ds["area_um"][val4]),
                          np.mean(ds["aspect"][val4]),
                          np.median(ds["aspect"][val4]),
                          np.std(ds["aspect"][val4]),
                          np.mean(ds["bright_avg"][val4]),
                          np.median(ds["bright_avg"][val4]),
                          np.std(ds["bright_avg"][val4]),
                          np.mean(ds["bright_sd"][val4]),
                          np.median(ds["bright_sd"][val4]),
                          np.std(ds["bright_sd"][val4]),                          
                          np.mean(ds["area_ratio"][val4]),
                          np.median(ds["area_ratio"][val4]),
                          np.std(ds["area_ratio"][val4]),
                          ratio1,
                          ratio2,
                          ratio3,
                          ratio4,
                          ratio5,
                          ratio6,
                          ratio7,
                          ratio8,
                          ratio9,
                          ratio10
                          ])    
    
    
all_parameters_mouse_rhoa = pd.DataFrame(all_parameters)

   
all_parameters_blind = []

for file in filepaths_blind:
    
##first area gate ##### EPCAM POSITIVE

    ds = dclab.new_dataset(file + ".rtdc")
    print(ds.config["setup"]["chip region"])

    ds.config["filtering"]["area_um min"] = 60
    ds.config["filtering"]["area_um max"] = 90
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["bright_avg min"] = 0
    ds.config["filtering"]["bright_avg max"] = 200
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val1 = ds.filter.all # valid events

## second area gate
    ds = dclab.new_dataset(file + ".rtdc")
    
    ds.config["filtering"]["area_um min"] = 60
    ds.config["filtering"]["area_um max"] = 90
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["bright_avg min"] = 0
    ds.config["filtering"]["bright_avg max"] = 200
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000

    ds.apply_filter() 
    val2 = ds.filter.all # valid events   
    
## third area gate
    ds = dclab.new_dataset(file + ".rtdc")
    
    ds.config["filtering"]["area_um min"] = 80
    ds.config["filtering"]["area_um max"] = 120
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["bright_avg min"] = 0
    ds.config["filtering"]["bright_avg max"] = 200
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val3 = ds.filter.all # valid events


## fourth area gate
    ds = dclab.new_dataset(file + ".rtdc")
    
    ds.config["filtering"]["area_um min"] = 120
    ds.config["filtering"]["area_um max"] = 400
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["bright_avg min"] = 0
    ds.config["filtering"]["bright_avg max"] = 200
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val4 = ds.filter.all # valid events
    
    

    ratio1 = 100*(sum(val1)/(sum(val1)+sum(val2)+sum(val3)+sum(val4)))
    ratio2 = 100*(sum(val2)/(sum(val1)+sum(val2)+sum(val3)+sum(val4)))
    ratio3 = 100*(sum(val3)/(sum(val1)+sum(val2)+sum(val3)+sum(val4)))
    ratio4 = 100*(sum(val4)/(sum(val1)+sum(val2)+sum(val3)+sum(val4)))
    ratio5 = 100*(sum(val1)/sum(val2))
    ratio6 = 100*(sum(val2)/sum(val3))
    ratio7 = 100*(sum(val3)/sum(val1))
    ratio8 = 100*(sum(val4)/sum(val1))
    ratio9 = 100*(sum(val4)/sum(val2))
    ratio10 = 100*(sum(val4)/sum(val3))
    
    all_parameters_blind.append([
                          np.mean(ds["deform"][val1]),
                          np.median(ds["deform"][val1]),
                          np.std(ds["deform"][val1]),
                          np.mean(ds["area_um"][val1]),
                          np.median(ds["area_um"][val1]),
                          np.std(ds["area_um"][val1]),
                          np.mean(ds["aspect"][val1]),
                          np.median(ds["aspect"][val1]),
                          np.std(ds["aspect"][val1]),
                          np.mean(ds["bright_avg"][val1]),
                          np.median(ds["bright_avg"][val1]),
                          np.std(ds["bright_avg"][val1]),
                          np.mean(ds["bright_sd"][val1]),
                          np.median(ds["bright_sd"][val1]),
                          np.std(ds["bright_sd"][val1]),                          
                          np.mean(ds["area_ratio"][val1]),
                          np.median(ds["area_ratio"][val1]),
                          np.std(ds["area_ratio"][val1]),
                          np.mean(ds["deform"][val2]),
                          np.median(ds["deform"][val2]),
                          np.std(ds["deform"][val2]),
                          np.mean(ds["area_um"][val2]),
                          np.median(ds["area_um"][val2]),
                          np.std(ds["area_um"][val2]),
                          np.mean(ds["aspect"][val2]),
                          np.median(ds["aspect"][val2]),
                          np.std(ds["aspect"][val2]),
                          np.mean(ds["bright_avg"][val2]),
                          np.median(ds["bright_avg"][val2]),
                          np.std(ds["bright_avg"][val2]),
                          np.mean(ds["bright_sd"][val2]),
                          np.median(ds["bright_sd"][val2]),
                          np.std(ds["bright_sd"][val2]),                          
                          np.mean(ds["area_ratio"][val2]),
                          np.median(ds["area_ratio"][val2]),
                          np.std(ds["area_ratio"][val2]),
                          np.mean(ds["deform"][val3]),
                          np.median(ds["deform"][val3]),
                          np.std(ds["deform"][val3]),
                          np.mean(ds["area_um"][val3]),
                          np.median(ds["area_um"][val3]),
                          np.std(ds["area_um"][val3]),
                          np.mean(ds["aspect"][val3]),
                          np.median(ds["aspect"][val3]),
                          np.std(ds["aspect"][val3]),
                          np.mean(ds["bright_avg"][val3]),
                          np.median(ds["bright_avg"][val3]),
                          np.std(ds["bright_avg"][val3]),
                          np.mean(ds["bright_sd"][val3]),
                          np.median(ds["bright_sd"][val3]),
                          np.std(ds["bright_sd"][val3]),                          
                          np.mean(ds["area_ratio"][val3]),
                          np.median(ds["area_ratio"][val3]),
                          np.std(ds["area_ratio"][val3]),
                          np.mean(ds["deform"][val4]),
                          np.median(ds["deform"][val4]),
                          np.std(ds["deform"][val4]),
                          np.mean(ds["area_um"][val4]),
                          np.median(ds["area_um"][val4]),
                          np.std(ds["area_um"][val4]),
                          np.mean(ds["aspect"][val4]),
                          np.median(ds["aspect"][val4]),
                          np.std(ds["aspect"][val4]),
                          np.mean(ds["bright_avg"][val4]),
                          np.median(ds["bright_avg"][val4]),
                          np.std(ds["bright_avg"][val4]),
                          np.mean(ds["bright_sd"][val4]),
                          np.median(ds["bright_sd"][val4]),
                          np.std(ds["bright_sd"][val4]),                          
                          np.mean(ds["area_ratio"][val4]),
                          np.median(ds["area_ratio"][val4]),
                          np.std(ds["area_ratio"][val4]),
                          ratio1,
                          ratio2,
                          ratio3,
                          ratio4,
                          ratio5,
                          ratio6,
                          ratio7,
                          ratio8,
                          ratio9,
                          ratio10
                          ])    
    
    
    
all_parameters_mouse_rhoa_blind = pd.DataFrame(all_parameters_blind)


figure = plt.figure(figsize=(20,20))
ax = plt.subplot(321, xlabel = 'Cell size [$\mu$m$^2$]', xlim = (10,200), ylabel = 'Deformation', ylim = (0, 0.15))
density_scatter(ds["area_um"], ds["deform"], bins = [1000,100], ax = ax)


#%%
################## PCA or LDA analysis

num_cat_1 = 16
num_cat_2 = 16 + num_cat_1
num_cat_3 = 15 + num_cat_2


xdata = all_parameters_mouse_rhoa
scaler = StandardScaler()
xdata = scaler.fit_transform(xdata)

ydata = np.ones(num_cat_3, dtype = int)
ydata[num_cat_1:num_cat_2] = 2
ydata[num_cat_2:num_cat_3] = 3

pca_a = 0
pca_b = 1
pca = PCA(n_components=2)
x_r = pca.fit_transform(xdata[0:num_cat_2])

### PCA loadings
eigenvectors = pd.DataFrame(pca.components_.T)
loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), columns = ['PC1', 'PC2']) 
most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.components_.shape[0])] 
sorted_importance = [np.abs(eigenvectors[i]).sort_values(axis=0, ascending=False) for i in range(pca.components_.shape[0])]
variances_per_component = pca.explained_variance_ratio_
variances_total = pca.explained_variance_ratio_.sum() * 100

for i in range(0,5):
    print(parameters[sorted_importance[0].index[i]])

labels = []
num_par = 30
for i in range(0,num_par):
    print(parameters[sorted_importance[0].index[i]])
    labels.append(parameters[sorted_importance[0].index[i]])

rect_scatter = [0.2, 0.3, 0.5, 0.5]
plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)
plt.scatter(np.arange(0, num_par, 1), sorted_importance[0][0:num_par])
ax.set_xlabel('Feature ')
ax.set_xticks(np.arange(0, num_par, 1))
ax.set_xticklabels(labels, rotation = 90)
ax.set_ylabel('Feature importance')


labels = []
num_par = 30
for i in range(0,num_par):
    print(parameters[sorted_importance[1].index[i]])
    labels.append(parameters[sorted_importance[1].index[i]])
  
   
plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)
plt.scatter(np.arange(0, num_par, 1), sorted_importance[1][0:num_par])
ax.set_xlabel('Feature ')
ax.set_xticks(np.arange(0, num_par, 1))
ax.set_xticklabels(labels, rotation = 90)
ax.set_ylabel('Feature importance')


plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)
plt.scatter( x_r[0:num_cat_1, pca_a], x_r[0:num_cat_1, pca_b], marker = ".", color = '#ab80d5', s = 130)
plt.scatter( x_r[num_cat_1:num_cat_2, pca_a], x_r[num_cat_1:num_cat_2, pca_b], marker = ".", color = '#1e8549', s = 130)
plt.scatter( x_r[num_cat_2:num_cat_3, pca_a], x_r[num_cat_2:num_cat_3, pca_b], marker = ".", color = 'orange', s = 130)           
ax.set_xlabel('PC1 ')
ax.set_ylabel(' PC2 ')
ax.tick_params(direction ='in')
ax.tick_params(direction ='in', width = 1, length = 4, which = 'both', grid_alpha = 0.2)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)
plt.gcf().set_tight_layout(False)



#%%

####################### BLIND EXPERIMENT  ####################### 
   
blind_data = scaler.transform(all_parameters_mouse_rhoa_blind)
blind_loading = pca.transform(blind_data)      
   
plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)
plt.scatter( x_r[0:num_cat_1, pca_a], x_r[0:num_cat_1, pca_b], marker = ".", color = '#ab80d5', s = 130)
plt.scatter( x_r[num_cat_1:num_cat_2, pca_a], x_r[num_cat_1:num_cat_2, pca_b], marker = ".", color = '#1e8549', s = 130)
plt.scatter( x_r[num_cat_2:num_cat_3, pca_a], x_r[num_cat_2:num_cat_3, pca_b], marker = ".", color = 'orange', s = 130) 
plt.scatter(blind_loading[0:15,0], blind_loading[0:15,1], marker = ".", color = 'blue', s = 130) 
for i in range(0,15):
    ax.annotate(i, (blind_loading[i][0], blind_loading[i][1]), fontsize = 10, color = 'blue') 
ax.set_xlabel('PC1 ')
ax.set_ylabel(' PC2 ')
ax.tick_params(direction ='in')
ax.tick_params(direction ='in', width = 1, length = 4, which = 'both', grid_alpha = 0.2)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)
plt.gcf().set_tight_layout(False)


#%%
############################## LOGISTIC REGRESSION #########################################

classifier_mouse = LogisticRegression(random_state = 0, penalty = 'l1', solver = 'liblinear') 
classifier_mouse.fit(x_r, ydata[0:num_cat_2]) 
probabilities_of_classes = classifier_mouse.predict_proba(x_r) 


X_set, y_set = x_r, ydata
plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 6,  
                               stop = X_set[:, 0].max() + 4, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 6,  
                               stop = X_set[:, 1].max() + 1, step = 0.01)) 
plt.contourf(X1, X2, classifier_mouse.predict( 
             np.array([X1.ravel(), X2.ravel()]).T).reshape( 
             X1.shape), alpha = 0.1, cmap = ListedColormap(('purple', 'green'))) 
plt.scatter( x_r[0:num_cat_1, pca_a], x_r[0:num_cat_1, pca_b], marker = ".", color = '#ab80d5', s = 130)
plt.scatter( x_r[num_cat_1:num_cat_2, pca_a], x_r[num_cat_1:num_cat_2, pca_b], marker = ".", color = '#1e8549', s = 130)
plt.scatter( x_r[num_cat_2:num_cat_3, pca_a], x_r[num_cat_2:num_cat_3, pca_b], marker = ".", color = 'orange', s = 130) 
ax.set_xlabel('PC1 ')
ax.set_ylabel(' PC2 ')
ax.tick_params(direction ='in')
ax.tick_params(direction ='in', width = 1, length = 4, which = 'both', grid_alpha = 0.2)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)
plt.gcf().set_tight_layout(False)



#%% 

############################## BOX PLOTS #########################################

all_parametersDF = pd.DataFrame(all_parameters, columns = parameters)

param = 'area_umME1'
#param = 'deformME1'
#param = 'aspect_ratioME1'
#param = 'area_ratioME1'
#param = 'area_umSTD1'
#param = 'deformSTD1'
#param = 'aspect_ratioSTD1'
#param = 'area_ratioSTD1'

ax = plt.axes()
objects = ('Control', 'Tumour')
y_pos = np.arange(len(objects))
data = [all_parametersDF[param][0:num_cat_1], all_parametersDF[param][num_cat_1:num_cat_2]]
plot_box_swarm(data, param)      
plt.xticks(y_pos, objects)
plt.ylabel('Median cell size [$\mu$m$^2$]')
plt.tick_params(direction ='in')
plt.subplots_adjust(wspace = 0.4, hspace = 0.4, top = 0.9, bottom = 0.2, left = 0.2, right = 0.8)
plt.rcParams["font.size"] = 20
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(2)
for axis in ['top', 'right']:
    ax.spines[axis].set_visible(False)
plt.tight_layout()
plt.show()


############################## STATISTICAL TEST #########################################

testres = wilcoxon(all_parametersDF[param][0:num_cat_1], all_parametersDF[param][num_cat_1:num_cat_2])
z = abs(norm.ppf(testres.pvalue/2))  #wilcoxon
effectsize = (abs(z)/math.sqrt(32))
print(testres, 'effect size is', effectsize)
              
                
#%%                 
##############################  PCA RANKING  #########################################


eigenvectors_abs = np.abs(eigenvectors[0].values)
data = {'60 - 90': eigenvectors_abs[0:12], '80 - 120': eigenvectors_abs[24:36], '120 - 400': eigenvectors_abs[36:48]}
index = ['Deformation mean', ' Deformation median', 'Deformation STD', 
                                               'Area mean', 'Area median', 'Area STD',
                                               'Aspect ratio mean', 'Aspect ratio  median', 'Aspect ratio STD',
                                               'Area ratio mean', ' Area ratio  median', 'Area ratio STD']
eigenvectors_df = pd.DataFrame (data, index = index)
plt.figure()
g = sns.heatmap(eigenvectors_df,  cmap = 'YlGnBu')
g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 12)
plt.rcParams["font.size"] = 12
plt.xlabel('Cell size [$\mu$m$^2$]')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


eigenvectors_abs = np.abs(eigenvectors[1].values)
data = {'60 - 90': eigenvectors_abs[0:12], '80 - 120': eigenvectors_abs[24:36], '120 - 400': eigenvectors_abs[36:48]}
index = ['Deformation mean', ' Deformation median', 'Deformation STD', 
                                               'Area mean', 'Area median', 'Area STD',
                                               'Aspect ratio mean', 'Aspect ratio  median', 'Aspect ratio STD',
                                               'Area ratio mean', ' Area ratio  median', 'Area ratio STD']
eigenvectors_df = pd.DataFrame (data, index = index)
plt.figure()
g = sns.heatmap(eigenvectors_df,  cmap = 'YlGnBu')
g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 12)
plt.rcParams["font.size"] = 12
plt.xlabel('Cell size [$\mu$m$^2$]')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()