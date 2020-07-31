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
import os
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
def read_timeSeries(transectName,plotTimeseries=False,resampling='none'):
    """
    Add description.
    
    Credits: Palash Patole, July 2020
    
    Arguments:
    -----------
    Add information here.
    Returns:
    -----------
    Add information here.
    
    """
    
    file_name = '../data/transect_time_series.csv'
    Transects = pd.read_csv(file_name,index_col='dates',parse_dates=True)
    Transects.drop('Unnamed: 0', axis =1, inplace = True)
    
    # Error handling for selected transect name
    pass

    # Creating a directory to store the results
    path_to_results = os.path.join(os.getcwd(), '../data/TSA_results/')
    if os.path.exists(path_to_results) == False:
        os.mkdir(path_to_results) 
    
    print("")
    print("******************************************************************")
    print("         Transect time series data analysis: inputs               ")
    print("******************************************************************")
    print("")

    # Printing the head of complete dataframe
    print("Head of the time series data: ")
    display(Transects.head())
    
    # Information about the selected transect
    print("\nTime series is selected for the transect: ",transectName)
    TimeSeries = Transects[transectName]
    # TimeSeries2 = Transects.ix[:,transectName]
    # print(type(TimeSeries),type(TimeSeries2))
    
    # Plotting the original time series for selected transect
    if plotTimeseries==True:
        
        title = "Original time series data for " + transectName
        ax = TimeSeries.plot(ls='--',marker = '*',c='r',lw = 1, ms = 6,title=title,figsize = (14,4),label='Original data')
        ax.legend()
        ax.grid()
        ax.set(xlabel='Year',ylabel='Shoreline location [m]')
        plt.savefig(path_to_results+'Orginal_time_series_'+transectName+'.png',transparent = False)
    
    # Linearly interpolating the time series
    print('{:.0f}% of the data in this original time series is missing.'.format(TimeSeries.isnull().sum() / len(TimeSeries) * 100))
    print('Linear interpolation is applied to find the missing data points.')
    TimeSeries_int = TimeSeries.interpolate() #linear interpolation between the data
    # display(TimeSeries_int.describe())
    # print(TimeSeries_int.isnull().sum())
    
    # Plotting the original and interpolated time series for selected transect
    if plotTimeseries==True:
        plt.figure()
        title = "Linearly interpolated time series data for " + transectName
        ax = TimeSeries.plot(ls='--',marker = '*',c='r',lw = 1, ms = 6,title=title,figsize = (14,4),label='Original data')
        TimeSeries_int.plot(ls='--',c='b',lw=1,label='Interpolated data')
        ax.set(xlabel='Year',ylabel='Shoreline location [m]')
        ax.legend()
        ax.grid()
        plt.savefig(path_to_results+'Interpolated_time_series_'+transectName+'.png',transparent = False)
    
    # Without resampling 
    if resampling=='none':
        return TimeSeries_int
    
    # For monthly average resampling 
    elif resampling=='MS':
        print('Data is resampled for average values on monthly basis and interpolated again.')
        TimerSeries_IR = TimeSeries_int.resample('MS').mean() # the interpolated data is resampled for Monthly intrevals
        # print(TimerSeries_IR.isnull().sum())
        TimeSeries_IRI = TimerSeries_IR.interpolate() # the Monthly resampled data are interpolated to avoid any NAN data point
        # print(TimerSeries_IRI.isnull().sum())
        
        # Plotting the original, interpolated and resampled-inteplorated time series data
        if plotTimeseries==True:
            plt.figure()
            title = "Resampled time series data for " + transectName
            ax = TimeSeries.plot(ls='--',marker = '*',c='r',lw = 1, ms = 6,title=title,figsize = (14,4),label='Original data')
            TimeSeries_int.plot(ls='--',c='b',lw=1,label='Interpolated data')
            TimeSeries_IRI.plot(marker='3',c='k', ms = 6, lw = 0.8, label = 'Interpolated resampled data')
            ax.set(xlabel='Year',ylabel='Shoreline location [m]')
            ax.legend()
            ax.grid()
            plt.savefig(path_to_results+'Resampled_time_series_'+transectName+'.png',transparent = False)
        
        return TimeSeries_IRI
    

#####################################################
########## Sesonal Decomposition ####################
#####################################################
def seasonal_decompose():
    pass

#####################################################
############# LOCAL TEST: REMOVE LATER ##############
#####################################################
transectName = "Transect 3"
Transects = read_timeSeries(transectName, plotTimeseries=True,resampling='MS')

# display(Transects.columns)