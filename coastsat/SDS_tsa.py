# Contributor: Palash Patole
# Updated on:  04 August 2020
#-------------------------------------------------------------------------------
""" This module performs the time-series analysis of a trasect's data and allows 
predicting the results into future.

To do:
    1. Based on Amin's work in a seperate jupyter notebook:
        a. [Done] Read the time series data from a .csv seperate file
           [To do] Error handling for the transect name
        b. Replace the above functionality by taking time series data as an input. 
           Provide an option to select a transect to analyze.
        c. [Done] After a/b, format the data.Linear intepolation, monthly resampling.
           [Done] Provide the option to plot(?)
        d. [Done] Provide the option to perform seasonal decomposition of the selected transect data
        e. [To do] Auto arima for the SARIMA
        f. [To do] ARIMA fitting
        g. [To do] Auto arima, GS, manual setting for the ARIMA
        h. [To do] Add values of RMSE and mean value of the series on predictions plot
        i. [To do] Add description for all classes and methods
        
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

# For computing accuracy of predictions
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse


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
def filePathGenerator():
    # Creating a directory to store the results
    path_to_results = os.path.join(os.getcwd(), '../data/TSA_results/')
    if os.path.exists(path_to_results) == False:
        os.mkdir(path_to_results) 

    return path_to_results

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
    
    print("")
    print("******************************************************************")
    print("         TSA: Reading a time series data for selected transect    ")
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
        plt.savefig(filePathGenerator()+'Orginal_time_series_'+transectName+'.png',transparent = False)
    
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
        plt.savefig(filePathGenerator()+'Interpolated_time_series_'+transectName+'.png',transparent = False)
    
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
            plt.savefig(filePathGenerator()+'Resampled_time_series_'+transectName+'.png',transparent = False)
        
        return TimeSeries_IRI
    

#####################################################
########## Sesonal Decomposition ####################
#####################################################
def seasonal_decompose(timeSeries,model='add',plotDecomposition=False):
    
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
    print("")
    print("******************************************************************")
    print("                TSA: Seasonal decomposition                       ")
    print("******************************************************************")
    print("")
    
    # Creating a directory to store the results
    path_to_results = os.path.join(os.getcwd(), '../data/TSA_results/')
    if os.path.exists(path_to_results) == False:
        os.mkdir(path_to_results) 
    
    # Decomposing the input time series
    decomposed = sm.tsa.seasonal_decompose(timeSeries,model=model)
    print("Seasonal decomposition of the input time series is successful.")
    
    # Plotting the decomposed time series 
    if plotDecomposition==True:
        matplotlib.rcParams['figure.figsize'] = 18, 8
        ax = decomposed.plot()
        plt.savefig(path_to_results+'Seasonal_decomposition_'+model+'.png', transparent=False)
        
        
    return decomposed


#%%
#####################################################
######### Class for time series forecast ############
#####################################################

class SDS_TSforecast:
    
    def __init__(self,TS,method):
        self.method = method
        self.TS = TS
        
    def setParameters(self,setting='auto_arima',pdq=0,seasonal_pdqm=0,printLogs=False): 
        print("")
        print("******************************************************************")
        print("        TSA: Setting parameters for the forecast model            ")
        print("******************************************************************")
        print("")
        
        # error handling 
        assert((setting=='GS') or (setting=='auto_arima')or (setting=='manual')),"Parameters of the model can not be set. Check the inputs for method: setParameters"
        
        self.__setting = setting
       
        
        if (setting=='auto_arima'):
            pass
        elif (setting=='GS'):
            # Error handling
            # check pdq and seasonal_pdqm are of the correct sizes
            pass
            
            if (self.method=='SARIMA'):
                AIC = []
                for param in pdq:
                    for param_seasonal in seasonal_pdqm:
                        try:
                            mod = sm.tsa.statespace.SARIMAX(self.TS,
                                                            order=param,
                                                            seasonal_order=param_seasonal,
                                                            enforce_stationarity=False,
                                                            enforce_invertibility=False)
                            results = mod.fit()
                
                            if (printLogs==True):
                                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                            
                            AIC.append([param, param_seasonal, results.aic])
                            
                        except:
                            continue
                AIC = sorted(AIC,key = lambda l:l[2])
                print("\nOptimal setting of parameters based on the AIC index:")
                print("(p,d,q)=",AIC[0][0])
                print("(P,D,Q,m)=",AIC[0][1])
                
                self.__pdq = AIC[0][0]
                self.__seasonal_pdqm = AIC[0][1]
                
                return(self.__pdq,self.__seasonal_pdqm)
        elif (setting=='manual'):
            if (self.method=='SARIMA'):
                self.__pdq = pdq
                self.__seasonal_pdqm = seasonal_pdqm
                
                print("\nInput parameters for",self.method ,"model are:")
                print("(p,d,q)=",self.__pdq)
                print("(P,D,Q,m)=",self.__seasonal_pdqm)
                
                return(self.__pdq,self.__seasonal_pdqm)
                    
    def fitmodel(self,splitPoint=0,validate=False,printSummary=False,plotPredictions=False):
        print("")
        print("******************************************************************")
        print("                 TSA: Fitting the forecast model                  ")
        print("******************************************************************")
        print("")
        
        
        # Error handling in case the train_test_split point has invalid value
        if ((splitPoint >= len(self.TS)) or (splitPoint==0)):
            splitPoint = len(self.TS)
            validate=False
            print("Model is being trained over the complete data.")
        
                
        # Spliting training and test data sets
        train = self.TS.iloc[:splitPoint]
        test = self.TS.iloc[splitPoint:]
        
        if (self.method=='SARIMA'):
            # Fitting the model to train data set
            self.__mod = sm.tsa.statespace.SARIMAX(train,
                                order=self.__pdq,
                                seasonal_order=self.__seasonal_pdqm,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            self.__results = self.__mod.fit()
            
            # Printing the results of model fitting process
            if (printSummary==True):
                print(self.__results.summary().tables[1])
                self.__results.plot_diagnostics(figsize=(16, 5))
                plt.show()
          
            # Validating the model predictions over test data set
            if validate == True:
                predictions = self.__results.get_prediction(start=splitPoint, end=len(self.TS)-1,typ='levels', dynamic=False)
                
                # Computing error in the predicted values
                MSE = mean_squared_error(test, predictions.predicted_mean)
                RMSE = rmse(test, predictions.predicted_mean)
                print("Mean squared error of the predictions is :", MSE)
                print("Root mean squared error of the predictions is :", RMSE)
                
                
                if (plotPredictions==True):
                    ax = plt.figure()
                    pred_ci = predictions.conf_int()
                    ax = test.plot(lw = 1,  color='b', label='Observed values')
                    predictions.predicted_mean.plot(ax=ax, label='Predicted values', color='r', lw =1, alpha=.7, figsize=(14, 7))
                    ax.fill_between(pred_ci.index,
                                    pred_ci.iloc[:, 0],
                                    pred_ci.iloc[:, 1], color='k', alpha=.2)
                    ax.set_ylabel('Shoreline location [m]') 
                    ax.set_xlabel('')
                    title = "Validating predictions with "+ self.method + " model"
                    ax.set_title(title)
                    plt.legend()
                    plt.grid()
                    plt.show()
                    plt.savefig(filePathGenerator()+"Predicted_time_series.png",transparent=False)
                    
        return self.__results
            

#####################################################
############# LOCAL TESTING: REMOVE LATER ###########
#####################################################
#%%
transectName = "Transect 3"
Transects = read_timeSeries(transectName, plotTimeseries=False,resampling='MS')

#%% 
# decomposition = seasonal_decompose(Transects,plotDecomposition=False)

# y2_trend = decomposition.trend
# y2_seasonal = decomposition.seasonal

# plt.figure(figsize=(16,5), dpi = 220)
# plt.plot(y2_trend, '--*r', lw = 0.5, ms = 4, label = 'Trend line')
# plt.plot(y2_seasonal, '--b', lw = 0.5, ms = 4, label = 'Seasonality line')
# plt.plot(Transects, 'k', ms = 2, lw = 0.5, label = 'Interpolated resampled data')
# plt.xlabel('Year')
# plt.ylabel('Shoreline location [m]')
# plt.legend()
# path_to_results = os.path.join(os.getcwd(), '../data/TSA_results/')
# plt.savefig(path_to_results+'Trend_seasonality.png', transparent = False)
# plt.show()

# display(Transects.columns)

#%%
Object = SDS_TSforecast(TS=Transects,method='SARIMA')

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 2) for x in list(itertools.product(p, d, q))]

# print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#%%
# temp1, temp2 = Object.setParameters(setting='GS',pdq=pdq,seasonal_pdqm=seasonal_pdq,printLogs=False)
Object.setParameters(setting='manual',pdq=(1,1,1),seasonal_pdqm=(1,1,1,70))


results= Object.fitmodel(splitPoint=376,validate=True,printSummary=True,plotPredictions=True)


#%%
