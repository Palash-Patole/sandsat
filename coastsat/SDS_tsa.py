# Contributor: Palash Patole
# Updated on:  29 July 2020
#-------------------------------------------------------------------------------
""" This module performs the time-series analysis of a trasect's data and allows 
predicting the results into future.

To do:
    1. Based on Amin's work in a seperate jupyter notebook:
        a. Read the time series data from a .csv seperate file
        b. Replace the above functionality by taking time series data as an input. 
           Provide an option to select a transect to analyze.
        c. After a/b, format the data.Linear intepolation, monthly resampling.
           Provide the option to plot(?)
        d. Provide the option to perform seasonal decomposition of the selected transect data
        ... 
        
"""

# Following package installation instruction is required, if it is absent from environment
# pip install statsmodels

#####################################################
########## Loading the required moduels #############
#####################################################

import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import warnings
import itertools
import numpy as np
warnings.filterwarnings("ignore")

# Following is specific to spyder IDE
from IPython.display import display

#####################################################
########## Setting parameters for plotting ##########
#####################################################

matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['text.color'] = 'k'
plt.rcParams['axes.facecolor'] = 'w'

#####################################################
########## Reading the time series data #############
#####################################################
def read_timeSeries(transectName,plotTimeseries=False):
    
    
    
    file_name = '../data/transect_time_series.csv'
    Transects = pd.read_csv(file_name,index_col='dates',parse_dates=True)
    Transects.drop('Unnamed: 0', axis =1, inplace = True)
    
    print("")
    print("******************************************************************")
    print("         Transect time series data analysis: inputs               ")
    print("******************************************************************")
    print("")
    
    print("Head of the time series data: ")

    display(Transects.head())
    
    TimeSeries = Transects[transectName]
    # TimeSeries = Transects.ix[:,'Transect 3']
    
    if plotTimeseries==True:
        
        title = "Original time series data for " + transectName
        ax = TimeSeries.plot(ls='--',marker = '*',c='r',lw = 1, ms = 6,title=title,figsize = (14,4),legend=True)
        ax.set(xlabel='',ylabel='Shoreline location [m]')
        # ax.savefig('Orginal_time_series_'+transectName+".png",transparent = False)
        
        filepath = '../data/'
        print(filepath)
        # fig.savefig(os.path.join(filepath,
                             # date + '_' + satname + '.jpg'), dpi=150)

        # plt.savefig('Original.png', transparent = False)
        # plt.show()
    
    return Transects
    

#####################################################
########## Sesonal Decomposition ####################
#####################################################
def seasonal_decompose():
    pass

#####################################################
############# LOCAL TEST: REMOVE LATER ##############
#####################################################
transectName = "Transect 3"
Transects = read_timeSeries(transectName, plotTimeseries=True)

display(Transects.columns)