# Contributor: Palash Patole
# Updated on:  04 August 2020
#-------------------------------------------------------------------------------
""" This module performs the time-series analysis of a trasect's data and allows 
predicting the results into future.

To do:
    1. Based on Amin's work in a seperate jupyter notebook:
        a. [Done] Read the time series data from a .csv seperate file
           [Done] Provide an option to select a transect to analyze.
           [To do] Error handling for the transect name
        b. Replace the above functionality by taking time series data as an input. 
        c. [Done] After a/b, format the data.Linear intepolation, monthly resampling.
           [Done] Provide the option to plot(?)
        d. [Done] Provide the option to perform seasonal decomposition of the selected transect data
        e. [To do] Auto arima for the SARIMA
        f. [To do] ARIMA fitting
        g. [To do] Auto arima, GS, manual setting for the ARIMA
        h. [To do] Add values of RMSE and mean value of the series on predictions plot - SARIMA and LSTM
        i. [To do] Add description for all classes and methods
        j. [Done] RNN based forecast method- initial setup
        k. [To do] RNN based forecast - when a model is imported, find the batch size
        l. [Done] Setting parameters of the LSTM model through setParameters method
        
"""

# Following package installation instruction is required, if it is absent from the environment
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

# For LSTM based model
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model


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
def read_timeSeries(transectName,plotTimeseries=False,resampling='none',readMethod='new'):
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
    if (readMethod=='new'): 
        file_name = '../data/transect_time_series.csv'
        Transects = pd.read_csv(file_name,index_col='dates',parse_dates=True)
        Transects.drop('Unnamed: 0', axis =1, inplace = True)
    elif (readMethod=='Old'): # Added 7th August to keep the old method of reading time-series used in Amin's notebook
        file_name = '../data/transect_time_series.csv'
        Transects = pd.read_csv(file_name)
        Transects['Date_Time'] = pd.to_datetime(Transects['dates'])
        Transects.set_index('Date_Time',inplace=True)
        Transects.drop('dates', axis =1, inplace = True)
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
    # TimeSeries = Transects.ix[:,transectName]
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
######### Class for time series analysis ############
#####################################################

class SDS_tsa:
    
    def __init__(self,TS,method):
        self.method = method
        self.TS = TS
        
        # error handling 
        assert((self.method=='SARIMA') or (self.method=='LSTM')),"The specified forecast method is not supported."
        
        
    def setParameters(self,Parasettings,setting='auto_arima',pdq=0,seasonal_pdqm=0,printLogs=False): 
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
                
            elif (self.method=='LSTM'):
                print("Grid serch technique of setting the parameters is not defined for the LSTM method.")
                
        elif (setting=='manual'):
            if (self.method=='SARIMA'):
                self.__pdq = pdq
                self.__seasonal_pdqm = seasonal_pdqm
                
                print("\nInput parameters for",self.method ,"model are:")
                print("(p,d,q)=",self.__pdq)
                print("(P,D,Q,m)=",self.__seasonal_pdqm)
                
            elif (self.method=='LSTM'):
                # Error handling 
                assert(type(Parasettings)==dict), "Settings should be passed as a variable of the dictionary type."
    
                # Extracting the settings, otherwise assigning the default values for some of the parameters
                self.__n_input = Parasettings["n_input"]
                self.__LSTM1_neurons = Parasettings["LSTM1_neurons"]
                self.__Nepochs = Parasettings["Nepochs"]
                
                try:    
                    self.__n_features = Parasettings["n_features"]    
                except:
                    self.__n_features = 1 #  always equal to one for a time series
                
                try:
                    self.__batches = Parasettings[ "batches"]
                except:
                    self.__batches = 1
                    
                try:
                    self.__activation_f1 = Parasettings["activation_f1"]
                except:
                    self.__activation_f1 = 'relu'
                    
                try:
                    self.__optimizer_type = Parasettings["optimizer_type"]
                except:
                    self.__optimizer_type = 'adam'
                    
                print("Input parameters for the ",self.method ,"model are:")                
                print("n_input=",self.__n_input)
                print("n_features=",self.__n_features)
                print("batches=",self.__batches)
                print("activation_f1=",self.__activation_f1)
                print("optimizer_type=",self.__optimizer_type)
                print("LSTM1_neurons=",self.__LSTM1_neurons)
                print("Nepochs=",self.__Nepochs)
                
    def loadModel(self,modelName = 'TSA_model'):
        print("")
        print("******************************************************************")
        print("                 TSA: Loading a forecast model                    ")
        print("******************************************************************")
        print("")
        
        # Full path to the model
        full_Path = filePathGenerator()+modelName+'.h5'
        
        # Error handling
        assert(os.path.exists(full_Path)),"No model exists with the specified name."
        
        # load the model
        if self.method=='LSTM':
            self.__mod = load_model(full_Path)
            print(modelName+ " model successfully loaded.")
            self.__mod.summary()
        else:
            print("Model can only be loaded for the LSTM method.")
            
    def fitmodel(self,splitPoint=0,validate=False,printSummary=False,plotPredictions=False,saveModel=False,modelName='TSA_model'):
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
                print("Mean value of the test data set is : ",test.mean())
                
                if (plotPredictions==True):
                    ax = plt.figure()
                    pred_ci = predictions.conf_int()
                    ax = test.plot(lw = 1,  color='b', label='Observed values')
                    predictions.predicted_mean.plot(ax=ax, label='Predicted values', color='r', lw =1, alpha=.7, figsize=(14, 7))
                    ax.fill_between(pred_ci.index,
                                    pred_ci.iloc[:, 0],
                                    pred_ci.iloc[:, 1], color='k', alpha=.2,label = 'Uncertainity')
                    ax.set_ylabel('Shoreline location [m]') 
                    ax.set_xlabel('')
                    title = "Validating predictions with "+ self.method + " model"
                    ax.set_title(title)
                    plt.legend()
                    plt.grid()
                    plt.show()
                    plt.savefig(filePathGenerator()+"Predicted_time_series.png",transparent=False)
                
                # return predictions alongwith results, if validated
                return self.__results, predictions
                
            return self.__results
        
        elif (self.method=='LSTM'):
            
            #Reshaping the data
            train2 = train.iloc[:].values.reshape(-1,1)
            if validate==True:
                test2 = test.iloc[:].values.reshape(-1,1)
            
            # Scaling and transforming the data - scalar fit only on the train data
            scaler = MinMaxScaler()
            scaler.fit(train2)
            scaled_train = scaler.transform(train2)
            if validate==True:
                scaled_test = scaler.transform(test2)
                            
            # Time series generator object
            train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=self.__n_input, batch_size=self.__batches)
            
            # define model
            self.__mod = Sequential()
            self.__mod.add(LSTM(self.__LSTM1_neurons, activation=self.__activation_f1, input_shape=(self.__n_input, self.__n_features)))
            self.__mod.add(Dense(1))
            self.__mod.compile(optimizer=self.__optimizer_type, loss='mse')
            self.__mod.summary()
                           
            # fit model
            self.__mod.fit_generator(train_generator,epochs=self.__Nepochs)   
            
            if (printSummary==True):
                loss_per_epoch = self.__mod.history.history['loss']
                plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
                plt.title('Fitting the LSTM based model over the scaled training dataset')
                plt.xlabel('# Epoch')
                plt.ylabel('Loss')
                plt.grid()
                
            # saving the model if required
            if (saveModel==True):
                self.__mod.save(filePathGenerator()+modelName+'.h5')
                print("The fitted model has been saved at :",filePathGenerator())
            
            # Validating the model predications over the test data
            if (validate == True):
                test_predictions = []
                first_eval_batch = scaled_train[-self.__n_input:]
                current_batch = first_eval_batch.reshape((self.__batches,self.__n_input,self.__n_features))
    
                for i in range(len(scaled_test)):
                    # predication one time stamp ahead 
                    current_pred = self.__mod.predict(current_batch)[0]
                    # store this prediction
                    test_predictions.append(current_pred)
                    # update the current batch to include this recent prediction and drop an old one 
                    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
                    
                # Inversing the scaling
                true_predictions = scaler.inverse_transform(test_predictions)
                
                # Creating a data frame holding predications and test data values
                Data = np.zeros((len(true_predictions),2))
                for index in range(len(true_predictions)):
                    Data[index,0] = test.iloc[index]
                    Data[index,1] = true_predictions[index]
                pred_df = pd.DataFrame(Data,test.index,['Original values','Predicted values'])
                
                # Computing error in the predicted values
                MSE = mean_squared_error(pred_df['Original values'], pred_df['Predicted values'])
                RMSE = rmse(pred_df['Original values'], pred_df['Predicted values'])
                print("Mean squared error of the predictions is :", MSE)
                print("Root mean squared error of the predictions is :", RMSE)
                print("Mean value of the test data set is : ",pred_df['Original values'].mean())
                
                if (plotPredictions==True):
                    fig, ax = plt.subplots()
                    pred_df.plot(ax=ax,figsize=(14, 7))
                    ax.set_ylabel('Shoreline location [m]') 
                    ax.set_xlabel('')
                    title = "Validating predictions with "+ self.method + " model"
                    ax.set_title(title)
                    plt.legend()
                    plt.grid()
                    plt.show()
                    plt.savefig(filePathGenerator()+"Predicted_time_series.png",transparent=False)
                
                # return predictions alongwith the model if validated:
                return self.__mod, pred_df
            
            return self.__mod
                    
    def forecast(self,steps=12,plotForecast=False):
        print("")
        print("******************************************************************")
        print("                 TSA: Forecasting with the chosen model           ")
        print("******************************************************************")
        print("")
        if (self.method=='SARIMA'):
            # Forecasting for a given number of steps
            pred_uc = self.__results.get_forecast(steps)
            print("Forecasted for ",steps," time steps in the future.")
            
            if plotForecast==True:
                plt.figure()
                pred_ci = pred_uc.conf_int()
                ax= self.TS.plot(label='Observed values', figsize=(14, 7), color = 'k', lw = 1)
                pred_uc.predicted_mean.plot(ax=ax, label='Forecasted values', color = 'r', lw = 1)
                ax.fill_between(pred_ci.index,
                                pred_ci.iloc[:, 0],
                                pred_ci.iloc[:, 1], color='m', alpha=.25, label = 'Uncertainity')
                ax.set_xlabel('')
                ax.set_ylabel('Shoreline location [m]')
                plt.legend(loc = 'upper left')
                plt.grid()
                plt.savefig(filePathGenerator()+'Forecasted_time_series.png', transparent = False)
                plt.show()
             
            return pred_uc
        
        elif (self.method=='LSTM'):
                if not hasattr(self, '__n_input'): # In case a saved model is used for the forecast
                    config = self.__mod.get_config()
                    layers = config['layers'][1]['config']['batch_input_shape']
                    self.__n_input = layers[1]
                    self.__batches = 1 # set as constant, to be extracted from the config
                    self.__n_features = 1
                    pass # Extract remaining parameters from the config
                
                # Creating train data and scaling it
                train = self.TS.iloc[:]
                train2 = train.iloc[:].values.reshape(-1,1) #Reshaping the data
                scaler = MinMaxScaler()
                scaler.fit(train2)
                scaled_train = scaler.transform(train2)
                
                # Forecasting
                forecasts = []
                first_eval_batch = scaled_train[-self.__n_input:]
                current_batch = first_eval_batch.reshape((self.__batches,self.__n_input,self.__n_features))
    
                for i in range(steps):
                    # predication one time stamp ahead 
                    current_foreC = self.__mod.predict(current_batch)[0]
                    # store this prediction
                    forecasts.append(current_foreC)
                    # update the current batch to include this recent prediction and drop an old one 
                    current_batch = np.append(current_batch[:,1:,:],[[current_foreC]],axis=1)
                    
                # Inversing the scaling
                true_forecasts = scaler.inverse_transform(forecasts)
                print("Forecasted for ",steps," time steps in the future.")
            
                # Creating a time series of the forecasted values
                last_timeStamp = self.TS.index[-1]
                freqTS = self.TS.index.freq
                indexForecast = pd.date_range(last_timeStamp, periods=steps+1, freq = freqTS) # Extract this frequency from self.TS
                indexForecast = indexForecast[1:]
                forecast_TS = pd.DataFrame(true_forecasts,indexForecast,['Forecasted values'])
                
                # Plotting the forecasted values
                if plotForecast==True:
                    plt.figure()
                    ax= self.TS.plot(label='Observed values', figsize=(14, 7), color = 'k', lw = 1)
                    forecast_TS.plot(ax=ax, label='Forecasted values', color = 'r', lw = 1)
                    ax.set_xlabel('')
                    ax.set_ylabel('Shoreline location [m]')
                    plt.legend(loc = 'upper left')
                    plt.grid()
                    plt.savefig(filePathGenerator()+'Forecasted_time_series.png', transparent = False)
                    plt.show()
                    
                return forecast_TS
            
            

#####################################################
############# LOCAL TESTING: REMOVE LATER ###########
#####################################################
#%%
transectName = "Transect 3"
Transects = read_timeSeries(transectName, plotTimeseries=False,resampling='MS',readMethod='Old')

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

#%% Multiple ways of forecasting
case = 3 # 1 - SARIMA based, grid search for para setting->fit->validate->fit over all data->forecast
         # 2 - SARIMA based, manual setting for parameters->fit->validate->fit over all data->forecast
         # 3 - LSTM based, manual setting for parameters->fit-validate->fit over all data->forecast
         # 4 - LSTM based, load fitted model -> forecast
         

if case==1:
# SARIMA based grid search for the parameters setting      
    # Creating an instance of the TSA class   
    Object = SDS_tsa(TS=Transects,method='SARIMA')

    # Setting parameters for the created instance
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 2) for x in list(itertools.product(p, d, q))]
    Object.setParameters(setting='GS',pdq=pdq,seasonal_pdqm=seasonal_pdq,printLogs=False)

    # Fit the model over the training data and validate against the test data
    Object.fitmodel(splitPoint=376,validate=True,printSummary=True,plotPredictions=True)
    
    # Re-training the model over the complete data and forecasting
    Object.fitmodel()
    Forecast_results = Object.forecast(steps=12,plotForecast=True)
    
elif case==2:
# SARIMA based, manual setting for the parameters    
    # Creating an instance of the TSA class   
    Object = SDS_tsa(TS=Transects,method='SARIMA')
    
    # Setting parameters for the created instance
    Object.setParameters(setting='manual',pdq=(1,1,1),seasonal_pdqm=(1,1,1,70))
    
    # Fit the model over the training data and validate against the test data
    Object.fitmodel(splitPoint=376,validate=True,printSummary=True,plotPredictions=True)
    
    # Re-training the model over the complete data and forecasting
    Object.fitmodel()
    Forecast_results = Object.forecast(steps=12,plotForecast=True)
    
elif case == 3:
# For the LSTM based model, manual setting for parameters to fit the model
    # Creating an instance of the TSA class     
    Object = SDS_tsa(TS=Transects,method='LSTM')
    
    # Setting the parameters for the model
    Paraset = {
    "n_input":12,
    "LSTM1_neurons": 100,
    "Nepochs":25
    }
    
    Object.setParameters(setting='manual',Parasettings = Paraset)

    # Fit the model over the training data and validate against the test data
    Object.fitmodel(splitPoint=376,validate=True,printSummary=True,plotPredictions=True)

    # # Re-training the model over the complete data and forecasting
    # Object.fitmodel(saveModel=True,modelName='TSA_model2') 
    # Forecast_results = Object.forecast(steps=12,plotForecast=True)

elif case==4:
# Loading a LSTM based model and forecasting using such a model
    # Creating an instance of the TSA class     
    Object = SDS_tsa(TS=Transects,method='LSTM')

    # Loading a saved model and then forecasting
    Object.loadModel(modelName='TSA_model2')
    fTS= Object.forecast(steps=12,plotForecast=True)



