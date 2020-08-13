# Contributor: Palash Patole
# Updated on:  12 August 2020
#-------------------------------------------------------------------------------
""" This is a script containing multiple use cases for the methods and classes 
defined in SDS_tsa.py 

"""
#%% Import the required modules
from coastsat import SDS_tsa

import matplotlib.pyplot as plt
import os
import itertools

# Following is specific to spyder IDE
from IPython.display import display


#%% Reading the time-series
transectName = "Transect 3"
Transects = SDS_tsa.read_timeSeries(transectName, plotTimeseries=False,resampling='MS',readMethod='Old')

if transectName=="Transect 1":
    Transects.dropna(inplace=True)

# #%% Seasonal decompose 
# decomposition = SDS_tsa.seasonal_decompose(Transects,plotDecomposition=False)

# y2_trend = decomposition.trend
# y2_seasonal = decomposition.seasonal

# plt.figure(figsize=(16,5), dpi = 220)
# plt.plot(y2_trend, '--*r', lw = 0.5, ms = 4, label = 'Trend line')
# plt.plot(y2_seasonal, '--b', lw = 0.5, ms = 4, label = 'Seasonality line')
# plt.plot(Transects, 'k', ms = 2, lw = 0.5, label = 'Interpolated resampled data')
# plt.xlabel('Year')
# plt.ylabel('Shoreline location [m]')
# plt.legend()
# path_to_results = os.path.join(os.getcwd(), 'data/TSA_results/')
# plt.savefig(path_to_results+'Trend_seasonality.png', transparent = False)
# plt.show()

#%% Multiple ways of forecasting
case = 3 # 1 - SARIMA based, grid search for para setting->fit->validate->fit over all data->forecast
         # 2 - SARIMA based, manual setting for parameters->fit->validate->fit over all data->forecast
         # 3 - LSTM based, manual setting for parameters->fit-validate->fit over all data->forecast
         # 4 - LSTM based, load fitted model -> forecast
         

if case==1:
# SARIMA based grid search for the parameters setting      
    # Creating an instance of the TSA class   
    Object = SDS_tsa.SDS_tsa(TS=Transects,method='SARIMA')

    # Setting parameters for the created instance
    p = d = q = range(0, 2)
    paraSet = {
    "pdq": list(itertools.product(p, d, q)),
    "seasonal_pdqm":[(x[0], x[1], x[2], 2) for x in list(itertools.product(p, d, q))]
    }    
    Object.setParameters(setting='GS',Parasettings=paraSet,printLogs=False)

    # Fit the model over the training data and validate against the test data
    Object.fitmodel(splitPoint=303,validate=True,printSummary=True,plotPredictions=True)
    
    # Re-training the model over the complete data and forecasting
    Object.fitmodel()
    Forecast_results = Object.forecast(steps=12,plotForecast=True)
    
elif case==2:
# SARIMA based, manual setting for the parameters    
    # Creating an instance of the TSA class   
    Object = SDS_tsa.SDS_tsa(TS=Transects,method='SARIMA')
    
    # Setting parameters for the created instance
    paraSet = {
    "pdq": (1,1,1),
    "seasonal_pdqm":(1,1,1,70)
    }  
    Object.setParameters(setting='manual',Parasettings=paraSet)
    
    # Fit the model over the training data and validate against the test data
    Object.fitmodel(splitPoint=376,validate=True,printSummary=True,plotPredictions=True)
    
    # Re-training the model over the complete data and forecasting
    Object.fitmodel()
    Forecast_results = Object.forecast(steps=12,plotForecast=True)
    
elif case == 3:
# For the LSTM based model, manual setting for parameters to fit the model
    # Creating an instance of the TSA class     
    Object = SDS_tsa.SDS_tsa(TS=Transects,method='LSTM')
    
    # Setting the parameters for the model
    Paraset = {
    "n_input":70,
    "LSTM1_neurons": 150,
    "Nepochs":50
    }
    
    Object.setParameters(setting='manual',Parasettings = Paraset)

    # Fit the model over the training data and validate against the test data
    Object.fitmodel(splitPoint=376,validate=True,printSummary=True,plotPredictions=True)

    # Re-training the model over the complete data and forecasting
    Object.fitmodel(saveModel=True,modelName='TSA_model5') 
    Forecast_results = Object.forecast(steps=12,plotForecast=True)

elif case==4:
# Loading a LSTM based model and forecasting using such a model
    # Creating an instance of the TSA class     
    Object = SDS_tsa.SDS_tsa(TS=Transects,method='LSTM')

    # Loading a saved model and then forecasting
    Object.loadModel(modelName='TSA_model2')
    fTS= Object.forecast(steps=12,plotForecast=True)



