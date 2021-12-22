# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:34:04 2020

@author: mkuban
"""

import numpy as np
import math
import dclab
import matplotlib.pylab as plt
import pandas as pd
from scipy.stats import pearsonr, mannwhitneyu
from numpy.polynomial.polynomial import polyfit
import seaborn as sns



def plot_box_swarm(data, parameter):

    plt.figure(figsize = (4,6))
    ax = plt.axes()
    y_pos = np.arange(0.5 , len(data), step = 2) # 4
      
    ax = sns.boxplot(data = data, palette = ["#49246d", "#1e8549"])
    sns.swarmplot(data = data, color = "0.25", size = 6)
    
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))

    plt.xticks(y_pos, labels = ['Healthy', 'Tumour'], rotation = 0)
    plt.ylabel(parameter) # ('Median cell size [$\mu$m$^2$]')
    plt.tick_params(direction ='in')
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4, top = 0.9, bottom = 0.2, left = 0.2, right = 0.8)

    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.5)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    plt.tight_layout()
    plt.show()



plt.rcParams['svg.fonttype'] = 'none'


#%%
############################################### FILE LOADING #######################################

filepaths = (
        r"Q:\Data\M001_data",
        # add filepaths
        )


all_parameters = []

for file in filepaths:
    
    ##first  gate
    ds = dclab.new_dataset(file + ".rtdc")
    ds.config["filtering"]["area_um min"] = 10
    ds.config["filtering"]["area_um max"] = 600
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.08
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    ds.apply_filter() 
    val1 = ds.filter.all # valid events ------------------------ all, control
    
    
    ## second  gate
    ds = dclab.new_dataset(file + ".rtdc")
    ds.config["filtering"]["area_um min"] = 10
    ds.config["filtering"]["area_um max"] = 600
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.08
    ds.config["filtering"]["fl1_max min"] = 220  
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    ds.apply_filter() 
    val2 = ds.filter.all # valid events ------------------------ CD45+
    
       
    ratio = 100*(sum(val2)/sum(val1))

    all_parameters.append([
                          np.mean(ds["deform"][val1]),
                          np.median(ds["deform"][val1]),
                          np.std(ds["deform"][val1]),                                        
                          np.mean(ds["area_um"][val1]),
                          np.median(ds["area_um"][val1]),
                          np.std(ds["area_um"][val1]),                           
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
                          np.mean(ds["area_ratio"][val2]),
                          np.median(ds["area_ratio"][val2]),
                          np.std(ds["area_ratio"][val2]),                          
                          np.mean(ds["aspect"][val2]),
                          np.median(ds["aspect"][val2]),
                          np.std(ds["aspect"][val2]), 
                          ratio])
    
all_parameters = pd.DataFrame(all_parameters, columns = ['Def_mean', 'Def_median', 'Def_STD', 
                                                                 'Area_mean', 'Area_median', 'Area_STD', 
                                                                 'Porosity_mean', 'Porosity_median', 'Porosity_STD', 
                                                                 'AR_mean', 'AR_median', 'AR_STD', 
                                                                 'Def_mean_CD45', 'Def_median_CD45', 'Def_STD_CD45',
                                                                 'Area_mean_CD45', 'Area_median_CD45', 'Area_STD_CD45', 
                                                                 'Porosity_mean_CD45', 'Porosity_median_CD45', 'Porosity_STD_CD45', 
                                                                 'AR_mean_CD45', 'AR_median_CD45', 'AR_STD_CD45', 
                                                                 'Ratio'])
 



#%%

################################ box plot Control vs TC CD45 % ################################ 

num_cat_1 = 6
num_cat_2 = num_cat_1 + 8

param = 'Def_median'
objects = ('Control\n(n=6)', 'TC\n(n=8)')
y_pos = np.arange(len(objects))
data = [all_parameters[param][0:num_cat_1], all_parameters[param][num_cat_1:num_cat_2]]

plot_box_swarm(data, param)   
plt.xticks(y_pos, objects)
plt.ylabel('Median deformation')
plt.tick_params(direction ='in')
plt.subplots_adjust(wspace = 0.4, hspace = 0.4, top = 0.9, bottom = 0.2, left = 0.4, right = 0.6)
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(2)
for axis in ['top', 'right']:
    ax.spines[axis].set_visible(False)
plt.tight_layout()
plt.show()


################################## statistical test ################################ 

testres = mannwhitneyu(all_parameters[param][0:num_cat_1], all_parameters[param][num_cat_1:num_cat_2])
z = (testres[0] - (6*8/2))/math.sqrt(6*8*15/12)
effectsize = (abs(z)/math.sqrt(14))
print(testres, 'Effect size is', effectsize)
            
                 
                 

#%%
################################ correlation of deformation and CD45 % ################################ 

   
x = all_parameters['Ratio']
ylabel = 'Def_median'
y = all_parameters[ylabel]
num_ct = 6
num_tc = num_ct + 8
    
### plot CD45 ratio and deformation

fig1=plt.figure(figsize = (10,10))
plot = plt.scatter(x[0:num_ct], y[0:num_ct], color='#ab80d5')
plot2 = plt.scatter(x[num_ct:num_tc], y[num_ct:num_tc], color='#1e8549')

a, b = polyfit(x, y,1)
x1 = np.arange(5, 70, 1)
plt.plot(x1, a+b*x1, '-', color='black')
plt.xlabel('% of CD45+ cells')
plt.ylabel(ylabel)
plt.tick_params(direction ='in')
plt.subplots_adjust(top = 0.7, bottom = 0.3, left = 0.3, right = 0.7)
plt.legend((plot, plot2),
           ('Control', 'Transfer colitis'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=16)
plt.rcParams["font.size"] = 20
plt.show()

pearson, pearson_pvalue = pearsonr(x,y)
print ('Pearson correlation ratio (Pearson r) is', pearson, 'with a p-value of ', pearson_pvalue)     
  

#%%
################################ plot Control vs TC Contours ################################ 


filepaths = (r"Q:\Data\Control_data")

for file in filepaths:
    
    ##first  gate
    ds = dclab.new_dataset(file + ".rtdc")
    ds.config["filtering"]["area_um min"] = 10
    ds.config["filtering"]["area_um max"] = 600
    ds.config["filtering"]["aspect min"] = 1
    ds.config["filtering"]["aspect max"] = 2
    ds.config["filtering"]["area_ratio min"] = 1
    ds.config["filtering"]["area_ratio max"] = 1.08
    ds.config["filtering"]["fl1_max min"] = -100
    ds.config["filtering"]["fl1_max max"] = 200000
    ds.config["filtering"]["fl2_max min"] = -100
    ds.config["filtering"]["fl2_max max"] = 200000
    ds.config["filtering"]["fl3_max min"] = -100
    ds.config["filtering"]["fl3_max max"] = 200000
    ds.apply_filter() 
    val1 = ds.filter.all # valid events ------------------------ all, control
    
    

filepaths = (r"Q:\Data\TC_data")

for file in filepaths:
    
    ##first  gate
    ds2 = dclab.new_dataset(file + ".rtdc")
    ds2.config["filtering"]["area_um min"] = 10
    ds2.config["filtering"]["area_um max"] = 600
    ds2.config["filtering"]["aspect min"] = 1
    ds2.config["filtering"]["aspect max"] = 2
    ds2.config["filtering"]["area_ratio min"] = 1
    ds2.config["filtering"]["area_ratio max"] = 1.08
    ds2.config["filtering"]["fl1_max min"] = -100
    ds2.config["filtering"]["fl1_max max"] = 200000
    ds2.config["filtering"]["fl2_max min"] = -100
    ds2.config["filtering"]["fl2_max max"] = 200000
    ds2.config["filtering"]["fl3_max min"] = -100
    ds2.config["filtering"]["fl3_max max"] = 200000
    ds2.apply_filter() 
    val2 = ds2.filter.all # valid events ------------------------ all, control
    
    
    ## second  gate
    ds2 = dclab.new_dataset(file + ".rtdc")
    ds2.config["filtering"]["area_um min"] = 10
    ds2.config["filtering"]["area_um max"] = 600
    ds2.config["filtering"]["aspect min"] = 1
    ds2.config["filtering"]["aspect max"] = 2
    ds2.config["filtering"]["area_ratio min"] = 1
    ds2.config["filtering"]["area_ratio max"] = 1.08
    ds2.config["filtering"]["fl1_max min"] = 220
    ds2.config["filtering"]["fl1_max max"] = 200000
    ds2.config["filtering"]["fl3_max min"] = -100
    ds2.config["filtering"]["fl3_max max"] = 200000
    ds2.apply_filter() 
    val3 = ds2.filter.all # valid events ------------------------ CD45+
    
       
left, width = 0.2, 0.5
bottom, height = 0.1, 0.5
spacing = 0.02
rect_scatter = [left, bottom, width, height]

plt.figure(figsize=(8, 8))
ax_scatter = plt.axes(rect_scatter)
    

ax_scatter.tick_params(direction='in', labelleft = True, labelbottom = True)
ax_scatter.set_xlim((0, 80))
ax_scatter.set_ylim((0, 0.1))
ax_scatter.set_xlabel('Cell size [$\mu$m$^2$]')
ax_scatter.set_ylabel('Deformation')
ax_scatter.tick_params(direction ='in')

sns.kdeplot(ds["area_um"][val1], ds["deform"][val1], levels = [0.5, 0.95, 1], fill = True, shade_lowest = False, cmap='Purples', alpha = 0.8, label = 'Control', ax = ax_scatter)
sns.kdeplot(ds2["area_um"][val2], ds2["deform"][val2], levels = [0.5, 0.95, 1], fill = True, shade_lowest = False, cmap='Greens', alpha = 0.8, label = 'TC', ax = ax_scatter)
sns.kdeplot(ds2["area_um"][val3], ds2["deform"][val3], levels = [0.5, 0.95, 1], fill = True, shade_lowest = False, cmap='Blues', alpha = 0.8, label = 'TC CD45+', ax = ax_scatter)

plt.rcParams["font.size"] = 20
for axis in ['top','bottom','left','right']:
    ax_scatter.spines[axis].set_linewidth(1)
for axis in ['top', 'right']:
    ax_scatter.spines[axis].set_visible(False)


plt.show()                     
                 