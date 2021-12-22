# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 08:27:10 2020

@author: mkuban
"""

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
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression 


# definitions for the axes
left, width = 0.2, 0.5
bottom, height = 0.1, 0.5
spacing = 0.02      
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.1]
rect_histy = [left + width + spacing, bottom, 0.1, height]


######## colormap chan be changed here
cmap_vir = cm.get_cmap('viridis')
cmap_new = cm.get_cmap('plasma_r')
cmap_blues = cm.get_cmap('Blues')

def density_scatter(x , y, bins, ax, sort = True, **kwargs )   :

    np.nan_to_num(y, copy=False)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    plt.figure(figsize=(8, 8))
    ax = plt.axes(rect_scatter)
    ax.tick_params(direction='in', labelleft = True, labelbottom = True)   
    plt.rcParams["font.size"] = 20
    ax.scatter( x, y, c=z, cmap = cmap_vir, marker = ".", s = 20, picker = True, **kwargs )    
#    sns.kdeplot(x, y, ax = ax, n_levels=10, cmap='Greys', alpha = 0.9)
    ax.set_xlim((10, 200))
    ax.set_ylim((0,0.2))
    ax.set_xlabel('Cell size [$\mu$m$^2$]')
    ax_scatter.set_ylabel('Deformation [a.u.]')
    ax.tick_params(direction ='in')
    ax.tick_params(direction ='in', width = 1, length = 4, which = 'both', grid_alpha = 0.2)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    plt.gcf().set_tight_layout(False)
    
    return ax


def density_scatter2( x , y, x2, y2, bins, ax, sort = True, **kwargs )   :

    np.nan_to_num(y, copy=False)
    xy = np.vstack([x2,y2])
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

    sns.kdeplot(x2, y2, n_levels=2, shade = True, shade_lowest = False, cmap='Greens', alpha = 0.4, label = 'Tumour', ax = ax_scatter)
    sns.kdeplot(x, y, n_levels=2, shade = True, shade_lowest = False, cmap = "Purples", alpha = 0.8, label = 'Healthy tissue', ax = ax_scatter)

# if we also want a scatter plot (single colour)
#    ax_scatter.scatter( x, y, marker = ".", color = '#ab80d5', s = 15, **kwargs) 
#    ax_scatter.scatter( x2, y2, marker = ".", color = '#1e8549',  s = 15, **kwargs) 
    
    ax_scatter.set_xlim((10, 200))
    ax_scatter.set_ylim((0, 0.2))
    ax_scatter.set_xlabel('Cell size [$\mu$m$^2$]')
    ax_scatter.set_ylabel('Deformation [a.u.]')
    ax_scatter.tick_params(direction ='in')
    ax_scatter.legend(loc = 'upper right', fontsize = 14)
   
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

    leg = ax_scatter.get_legend()
    leg.legendHandles[0].set_color('#5c9e77')
    leg.legendHandles[1].set_color('#ab80d5')

                     
    plt.rcParams["font.size"] = 18
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.gcf().set_tight_layout(False)
    
    return ax


def density_scatter3( x , y, x2, y2, bins, ax, sort = True, **kwargs )   :

    np.nan_to_num(y, copy=False)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

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
#    sns.kdeplot(x, y, ax = ax_scatter, n_levels=10, cmap='Oranges', alpha = 0.9)
    ax_scatter.set_xlim((10, 200))
    ax_scatter.set_ylim((80, 150))
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

#%%
    
### set parameters
    
parameters = [
                          "deformME1",
                          "deformMD1",
                          "deformSTD1",
                          "area_umME1",
                          "area_umMD1",
                          "area_umSTD1",
#                          "bright_avgME1",
#                          "bright_avgMD1",
#                          "bright_avgSTD1",
#                          "bright_sdME1",
#                          "bright_sdMD1",
#                          "bright_sdSTD1",                        
                          "area_ratioME1",
                          "area_ratioMD1",
                          "area_ratioSTD1",
                          "aspect_ratioME1",
                          "aspect_ratioMD1",
                          "aspect_ratioSTD1",
                          "deformME2",
                          "deformMD2",
                          "deformSTD2",
                          "area_umME2",
                          "area_umMD2",
                          "area_umSTD2",
#                          "bright_avgME2",
#                          "bright_avgMD2",
#                          "bright_avgSTD2",
#                          "bright_sdME2",
#                          "bright_sdMD2",
#                          "bright_sdSTD2",                        
                          "area_ratioME2",
                          "area_ratioMD2",
                          "area_ratioSTD2",
                          "aspect_ratioME2",
                          "aspect_ratioMD2",
                          "aspect_ratioSTD2",
                          "deformME3",
                          "deformMD3",
                          "deformSTD3",
                          "area_umME3",
                          "area_umMD3",
                          "area_umSTD3",
#                          "bright_avgME3",
#                          "bright_avgMD3",
#                          "bright_avgSTD3",
#                          "bright_sdME3",
#                          "bright_sdMD3",
#                          "bright_sdSTD3",                        
                          "area_ratioME3",
                          "area_ratioMD3",
                          "area_ratioSTD3",
                          "aspect_ratioME3",
                          "aspect_ratioMD3",
                          "aspect_ratioSTD3",
                          "deformME4",
                          "deformMD4",
                          "deformSTD4",
                          "area_umME4",
                          "area_umMD4",
                          "area_umSTD4",
#                          "bright_avgME4",
#                          "bright_avgMD4",
#                          "bright_avgSTD4",
#                          "bright_sdME4",
#                          "bright_sdMD4",
#                          "bright_sdSTD4",                        
                          "area_ratioME4",
                          "area_ratioMD4",
                          "area_ratioSTD4",
                          "aspect_ratioME4",
                          "aspect_ratioMD4",
                          "aspect_ratioSTD4",
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
################################################ FILE LOADING #######################################

filepaths = (
    r"Q:\Data\RTDC\M001",
    # add filepaths - first a list of healthy, then corresponding tumour samples
    )

all_parameters = []

for file in filepaths:
    
##first area gate
    ds = dclab.new_dataset(file + ".rtdc")
    sample = 'Surrounding tissue'
    
    ds.config["filtering"]["area_um min"] = 20
    ds.config["filtering"]["area_um max"] = 50
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val1 = ds.filter.all # valid events

## second area gate
    ds = dclab.new_dataset(file + ".rtdc")
    sample = 'Surrounding tissue'
    
    ds.config["filtering"]["area_um min"] = 20
    ds.config["filtering"]["area_um max"] = 50
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000

    ds.apply_filter() 
    val2 = ds.filter.all # valid events   
    
## third area gate
    ds = dclab.new_dataset(file + ".rtdc")
    sample = 'Surrounding tissue'
    
    ds.config["filtering"]["area_um min"] = 50
    ds.config["filtering"]["area_um max"] = 600
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val3 = ds.filter.all # valid events


## third area gate
    ds = dclab.new_dataset(file + ".rtdc")
    sample = 'Surrounding tissue'
    
    ds.config["filtering"]["area_um min"] = 50
    ds.config["filtering"]["area_um max"] = 600
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
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
#                          np.mean(ds["bright_avg"][val1]),
#                          np.median(ds["bright_avg"][val1]),
#                          np.std(ds["bright_avg"][val1]),
#                          np.mean(ds["bright_sd"][val1]),
#                          np.median(ds["bright_sd"][val1]),
#                          np.std(ds["bright_sd"][val1]),                          
                          np.mean(ds["area_ratio"][val1]),
                          np.median(ds["area_ratio"][val1]),
                          np.std(ds["area_ratio"][val1]),
                          np.mean(ds["aspect"][val1]),
                          np.median(ds["aspect"][val1]),
                          np.std(ds["aspect"][val1]),
                          np.mean(ds["deform"][val2]),
                          np.median(ds["deform"][val2]),
                          np.std(ds["deform"][val2]),
                          np.mean(ds["area_um"][val2]),
                          np.median(ds["area_um"][val2]),
                          np.std(ds["area_um"][val2]),
#                          np.mean(ds["bright_avg"][val2]),
#                          np.median(ds["bright_avg"][val2]),
#                          np.std(ds["bright_avg"][val2]),
#                          np.mean(ds["bright_sd"][val2]),
#                          np.median(ds["bright_sd"][val2]),
#                          np.std(ds["bright_sd"][val2]),                          
                          np.mean(ds["area_ratio"][val2]),
                          np.median(ds["area_ratio"][val2]),
                          np.std(ds["area_ratio"][val2]),
                          np.mean(ds["aspect"][val2]),
                          np.median(ds["aspect"][val2]),
                          np.std(ds["aspect"][val2]),
                          np.mean(ds["deform"][val3]),
                          np.median(ds["deform"][val3]),
                          np.std(ds["deform"][val3]),
                          np.mean(ds["area_um"][val3]),
                          np.median(ds["area_um"][val3]),
#                          np.std(ds["area_um"][val3]),
#                          np.mean(ds["bright_avg"][val3]),
#                          np.median(ds["bright_avg"][val3]),
#                          np.std(ds["bright_avg"][val3]),
#                          np.mean(ds["bright_sd"][val3]),
#                          np.median(ds["bright_sd"][val3]),
#                          np.std(ds["bright_sd"][val3]),                          
                          np.mean(ds["area_ratio"][val3]),
                          np.median(ds["area_ratio"][val3]),
                          np.std(ds["area_ratio"][val3]),
                          np.mean(ds["aspect"][val3]),
                          np.median(ds["aspect"][val3]),
                          np.std(ds["aspect"][val3]),
                          np.mean(ds["deform"][val4]),
                          np.median(ds["deform"][val4]),
                          np.std(ds["deform"][val4]),
                          np.mean(ds["area_um"][val4]),
                          np.median(ds["area_um"][val4]),
                          np.std(ds["area_um"][val4]),
#                          np.mean(ds["bright_avg"][val4]),
#                          np.median(ds["bright_avg"][val4]),
#                          np.std(ds["bright_avg"][val4]),
#                          np.mean(ds["bright_sd"][val4]),
#                          np.median(ds["bright_sd"][val4]),
#                          np.std(ds["bright_sd"][val4]),                          
                          np.mean(ds["area_ratio"][val4]),
                          np.median(ds["area_ratio"][val4]),
                          np.std(ds["area_ratio"][val4]),
                          np.mean(ds["aspect"][val4]),
                          np.median(ds["aspect"][val4]),
                          np.std(ds["aspect"][val4]),
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

#%%
#################################### PCA ###########################################

all_test = pd.DataFrame( all_parameters)
xdata= all_test
scaler = MinMaxScaler()
scaler.fit(xdata)
xdata= scaler.transform(xdata)

pca = PCA(n_components=2)
x_r = pca.fit_transform(xdata)
eigenvectors = pd.DataFrame(pca.components_.T)
loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), columns = ['PC1', 'PC2'])
most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca.components_.shape[0])] ### most important variable index (for pc1, pc2)
sorted_importance = [np.abs(eigenvectors[i]).sort_values(axis=0, ascending=False) for i in range(pca.components_.shape[0])]
variances_per_component = pca.explained_variance_ratio_
variances_total = pca.explained_variance_ratio_.sum() * 100


num = 30
labels = []
for i in range(0,num):
    print(parameters[sorted_importance[1].index[i]])
    labels.append(parameters[sorted_importance[1].index[i]])

rect_scatter = [0.2, 0.3, 0.5, 0.5]
plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)
plt.scatter(np.arange(0, num, 1), sorted_importance[1][0:num])
ax.set_xlabel('Feature ')
ax.set_xticks(np.arange(0, num, 1))
ax.set_xticklabels(labels, fontsize = 9, rotation = 90)
ax.set_ylabel('Feature importance')
plt.show()


num_cat_1 = 15
num_cat_2 = 15 + num_cat_1

### plot the PCAs
plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)

plt.scatter( x_r[0:num_cat_1,0], x_r[0:num_cat_1,1], marker = ".", color = '#ab80d5', s = 130) 
plt.scatter( x_r[num_cat_1:num_cat_2,0], x_r[num_cat_1:num_cat_2,1], marker = ".", color = '#1e8549', s = 130)           
ax.set_xlabel('PC1 ')
ax.set_ylabel(' PC2 ')
ax.tick_params(direction ='in')
ax.tick_params(direction ='in', width = 1, length = 4, which = 'both', grid_alpha = 0.2)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)
plt.gcf().set_tight_layout(False)




#%%
####################################### LOGISTIC REGRESSION ###########################################
plt.rcParams["font.size"] = 20

yfit = np.array([1, 1, 1, ...]) # enter 1 and 2 according to data labels (healthy or tumour)
xfit = x_r
classifier = LogisticRegression(random_state = 0) 
classifier.fit(xfit, yfit) 

X_set, y_set = xfit, yfit
plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,  
                               stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1,  
                               stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict( 
             np.array([X1.ravel(), X2.ravel()]).T).reshape( 
             X1.shape), alpha = 0.1, cmap = ListedColormap(('purple', 'green'))) 
  

### plot the PCAs

plt.scatter( x_r[0:num_cat_1,0], x_r[0:num_cat_1,1], marker = ".", color = '#ab80d5', s = 130) 
plt.scatter( x_r[num_cat_1:num_cat_2,0], x_r[num_cat_1:num_cat_2,1], marker = ".", color = '#1e8549', s = 130) 

plt.scatter( x_r[num_cat_2:num_cat_3,0], x_r[num_cat_2:num_cat_3,1], marker = ".", color = 'blue', s = 130)        # lime green 
plt.scatter( x_r[num_cat_3:num_cat_4,0], x_r[num_cat_3:num_cat_4,1], marker = ".", color = 'blue', s = 130)        # indigo
            

ax.set_xlabel('PC1 ')
ax.set_ylabel(' PC2 ')
ax.tick_params(direction ='in')
ax.tick_params(direction ='in', width = 1, length = 4, which = 'both', grid_alpha = 0.2)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)
plt.gcf().set_tight_layout(False)
plt.show()



#%%
############################################### PCA test #######################################

filepaths = (
    r"S:\Data\RTDC\M001",
    # add filepaths
        )

all_params = []

for file in filepaths:
    
##first area gate
    ds = dclab.new_dataset(file + ".rtdc")
    sample = 'Surrounding tissue'
    
    ds.config["filtering"]["area_um min"] = 60
    ds.config["filtering"]["area_um max"] = 100
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val1 = ds.filter.all # valid events

## second area gate
    ds = dclab.new_dataset(file + ".rtdc")
    sample = 'Surrounding tissue'
    
    ds.config["filtering"]["area_um min"] = 60
    ds.config["filtering"]["area_um max"] = 100
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000

    ds.apply_filter() 
    val2 = ds.filter.all # valid events   
    
## third area gate
    ds = dclab.new_dataset(file + ".rtdc")
    sample = 'Surrounding tissue'
    
    ds.config["filtering"]["area_um min"] = 100
    ds.config["filtering"]["area_um max"] = 200
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    
    ds.apply_filter() 
    val3 = ds.filter.all # valid events


## third area gate
    ds = dclab.new_dataset(file + ".rtdc")
    sample = 'Surrounding tissue'
    
    ds.config["filtering"]["area_um min"] = 200
    ds.config["filtering"]["area_um max"] = 600
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.1
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
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
    
    all_params.append([
                          np.mean(ds["deform"][val1]),
                          np.median(ds["deform"][val1]),
                          np.std(ds["deform"][val1]),
                          np.mean(ds["area_um"][val1]),
                          np.median(ds["area_um"][val1]),
                          np.std(ds["area_um"][val1]),
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
                          ratio10])

## PCA
    
xtestdata = pd.DataFrame( all_params)
xtestdata= scaler.transform(xtestdata)
xtest_r = pca.transform(xtestdata)

num_cat_1_test = 5
num_cat_2_test = 5 + num_cat_1


plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)

### plot the PCAs
plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)

plt.scatter( x_r[0:num_cat_1,0], x_r[0:num_cat_1,1], marker = ".", color = '#ab80d5', s = 130) 
plt.scatter( x_r[num_cat_1:num_cat_2,0], x_r[num_cat_1:num_cat_2,1], marker = ".", color = '#1e8549', s = 130) 
plt.scatter( xtest_r[0:num_cat_1_test,0], xtest_r[0:num_cat_1_test,1], marker = ".", color = '#4B0082', s = 130)        # indigo
plt.scatter( xtest_r[num_cat_1_test:num_cat_2_test,0], xtest_r[num_cat_1_test:num_cat_2_test,1], marker = ".", color = '#32CD32', s = 130)  # lime green

ax.set_xlabel('PC1 ')
ax.set_ylabel(' PC2 ')
ax.tick_params(direction ='in')
ax.tick_params(direction ='in', width = 1, length = 4, which = 'both', grid_alpha = 0.2)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)
plt.gcf().set_tight_layout(False)
plt.show()

#%%
######################### TEST LOGISTIC REGRESSION ###########################################

xdata = pd.DataFrame( all_parameters)
scaler = MinMaxScaler()
scaler.fit(xdata)
xdata= scaler.transform(xdata)

pca = PCA(n_components=2)
x_for_prediction = pca.fit_transform(xdata)

ytest = np.array([1, 1, 1, 1, 2, 2, 2, 2]) # here 4 healthy and 4 tumour samples 
y_pred = classifier.predict(x_for_prediction) 


## plotting
X_set, y_set = xfit, yfit
plt.figure(figsize=(8, 8))
ax = plt.axes(rect_scatter)

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,  
                               stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1,  
                               stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict( 
             np.array([X1.ravel(), X2.ravel()]).T).reshape( 
             X1.shape), alpha = 0.1, cmap = ListedColormap(('red', 'green'))) 
            
plt.scatter( x_for_prediction[0:4, 0], x_for_prediction[0:4, 1], marker = "x", color = 'black', s = 130) 
plt.scatter( x_for_prediction[4:8, 0], x_for_prediction[4:8, 1], marker = "x", color = 'brown', s = 130) 

ax.set_xlabel('PC1 ')
ax.set_ylabel(' PC2 ')
ax.tick_params(direction ='in')
ax.tick_params(direction ='in', width = 1, length = 4, which = 'both', grid_alpha = 0.2)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)
plt.gcf().set_tight_layout(False)
plt.show()



#%%
############## Table of importance of features


eigenvectors_abs = np.abs(eigenvectors[0].values)
data = {'20-50': eigenvectors_abs[0:12], '50-600': eigenvectors_abs[24:36]}
index = parameters[0:12]
eigenvectors_df = pd.DataFrame (data, index = index)
plt.figure()
g = sns.heatmap(eigenvectors_df,  cmap = 'YlGnBu')
#g = sns.heatmap(eigenvectors_df,  cmap = 'viridis')
g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 12)
plt.rcParams["font.size"] = 12
plt.xlabel('Cell size [$\mu$m$^2$]')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


eigenvectors_abs = np.abs(eigenvectors[1].values)
data = {'20-50': eigenvectors_abs[0:12], '50-600': eigenvectors_abs[24:36]}
index = parameters[0:12]
eigenvectors_df = pd.DataFrame (data, index = index)
plt.figure()
g = sns.heatmap(eigenvectors_df,  cmap = 'YlGnBu')
#g = sns.heatmap(eigenvectors_df,  cmap = 'viridis')
g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 12)
plt.rcParams["font.size"] = 12
plt.xlabel('Cell size [$\mu$m$^2$]')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()