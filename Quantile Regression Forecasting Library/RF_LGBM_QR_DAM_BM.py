import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta as td
from datetime import datetime
# from pandas import Timedelta as td
import traceback
# from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn_quantile import RandomForestQuantileRegressor
import warnings

    
def generate_train_and_test_dataframes_RF_DAM(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                        test_start_time: dt, test_end_time: dt):

    train_X = None
    train_y = None 
    test_X = None 
    test_y = None 
    test_df = None

    try:
        
        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
            return train_X, train_y, test_X, test_y, test_df
        
        original_columns = list(participant_df.columns)

        participant_df = participant_df.dropna()

        date_format = "%d/%m/%Y %H:%M"

        train_df = None
        

        
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)    
        train_df = participant_df[(participant_df.index>=train_start_time_str) & (participant_df.index<train_end_time_str)].copy(deep="True")

        if train_df is None or len(train_df) == 0:
            print("Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df


        
        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)    
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format) 
        test_df = participant_df[(participant_df.index>=test_start_time_str) & (participant_df.index<test_end_time_str)].copy(deep="True")

        if test_df is None or len(test_df) == 0:
            print("Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df


        train_X = train_df.iloc[:, 24:]
        test_X = test_df.iloc[:, 24:]
        train_y = train_df.iloc[:, 0:24]
        test_y = test_df.iloc[:, 0:24]
                                
        return train_X, train_y, test_X, test_y, test_df
        
    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df
    
    
def fit_multitarget_model_RF_DAM(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):
  
    try:
        
#         model.fit(X_train, Y_train) if len(targets) > 1 else model.fit(X_train, Y_train.values.ravel())  
        model.fit(X_train, Y_train) 

        model_test_predictions=None  
        model_test_predictions = model.predict(X_test).reshape(5,24)                            
        cols = Y_test.iloc[:, 0:24].columns.values.tolist() 
        
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_10"] = model_test_predictions[0:1,i].tolist() if len(cols) > 1 else model_test_predictions.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_30"] = model_test_predictions[1:2,i].tolist() if len(cols) > 1 else model_test_predictions.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_50"] = model_test_predictions[2:3,i].tolist() if len(cols) > 1 else model_test_predictions.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_70"] = model_test_predictions[3:4,i].tolist() if len(cols) > 1 else model_test_predictions.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_90"] = model_test_predictions[4:,i].tolist() if len(cols) > 1 else model_test_predictions.tolist()
            

        
        
        
           
        return actuals_and_forecast_df
    
    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()
    
def rolling_walk_forward_validation_RF_DAM(model, data, targets, start_time, end_time, training_days, path):
 
    try:

        all_columns = list(data.columns)            
        results = pd.DataFrame()
            

        date_format = "%d/%m/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)
        
        while start_time < end_time:
            
            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time 
    
            test_start_time = train_end_time + td(hours=24)
            test_end_time = test_start_time + td(hours=1)
            
            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))
    
            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes_RF_DAM(participant_df=data, train_start_time=train_start_time.strftime("%d/%m/%Y %H:%M"), train_end_time=train_end_time.strftime("%d/%m/%Y %H:%M"), 
                            test_start_time=test_start_time.strftime("%d/%m/%Y %H:%M"), test_end_time=test_end_time.strftime("%d/%m/%Y %H:%M"))
            
            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
    
            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
            
            actuals_and_forecast_df = fit_multitarget_model_RF_DAM(model=model, X_train=train_X, Y_train=train_y,
                                            X_test=test_X, Y_test=test_y, actuals_and_forecast_df=test_df,targets=test_df.iloc[:,0:24].columns.values.tolist())
    
            results = pd.concat([results, actuals_and_forecast_df])
            start_time = start_time + td(hours=24)
            
        results.to_csv(path  + ".csv", index = False)
        
          
        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()    
    
    


def load_data_DAM(file_path):
    # Suppress warnings and display all columns
    warnings.filterwarnings('ignore')
    pd.set_option("display.max_columns", None)
    
    # Define date format and parsing function
    date_format = "%d/%m/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    
    # Read the CSV file
    dat = pd.read_csv(file_path, index_col="DeliveryPeriod", parse_dates=True, date_parser=date_parse)
    
    # Handle missing values
    dat = dat.bfill(axis='rows')
    dat = dat.ffill(axis='rows')
    
    # Select target columns
    Y = dat.iloc[:, 0:24]
    
    return dat, Y


    
    
def generate_train_and_test_dataframes_RF_BM(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                        test_start_time: dt, test_end_time: dt):

  
    train_X = None
    train_y = None 
    test_X = None 
    test_y = None 
    test_df = None

    try:
        
        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
            return train_X, train_y, test_X, test_y, test_df
        
        original_columns = list(participant_df.columns)

        participant_df = participant_df.dropna()

        date_format="%m/%d/%Y %H:%M"

        train_df = None
        

        
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)    
        train_df = participant_df[(participant_df.index>=train_start_time_str) & (participant_df.index<train_end_time_str)].copy(deep="True")

        if train_df is None or len(train_df) == 0:
            print("Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df


        
        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)    
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format) 
        test_df = participant_df[(participant_df.index>=test_start_time_str) & (participant_df.index<test_end_time_str)].copy(deep="True")

        if test_df is None or len(test_df) == 0:
            print("Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df


        train_X = train_df.iloc[:, 16:]
        test_X = test_df.iloc[:, 16:]
        train_y = train_df.iloc[:, 0:16]
        test_y = test_df.iloc[:, 0:16]
                                
        return train_X, train_y, test_X, test_y, test_df
        
    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df
    
    
def fit_multitarget_model_RF_BM(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):
  
    try:
        
        model.fit(X_train, Y_train) 

        model_test_predictions=None  
        model_test_predictions = model.predict(X_test).reshape(5,16)                            
        cols = Y_test.iloc[:, 0:16].columns.values.tolist()  
        
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_10"] = model_test_predictions[0:1,i].tolist() if len(cols) > 1 else model_test_predictions.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_30"] = model_test_predictions[1:2,i].tolist() if len(cols) > 1 else model_test_predictions.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_50"] = model_test_predictions[2:3,i].tolist() if len(cols) > 1 else model_test_predictions.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_70"] = model_test_predictions[3:4,i].tolist() if len(cols) > 1 else model_test_predictions.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_90"] = model_test_predictions[4:,i].tolist() if len(cols) > 1 else model_test_predictions.tolist()
           
        return actuals_and_forecast_df
    
    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()
    
def rolling_walk_forward_validation_RF_BM(model, data, targets, start_time, end_time, training_days, path):
 
    try:

        all_columns = list(data.columns)            
        results = pd.DataFrame()
            

        date_format="%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)
        
        while start_time < end_time:
            
            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time 
    
            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(minutes=30)
            
            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))
    
            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes_RF_BM(participant_df=data, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"), train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"), 
                            test_start_time=test_start_time.strftime("%m/%d/%Y %H:%M"), test_end_time=test_end_time.strftime("%m/%d/%Y %H:%M"))
            
            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
    
            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
            
            actuals_and_forecast_df = fit_multitarget_model_RF_BM(model=model, X_train=train_X, Y_train=train_y,
                                            X_test=test_X, Y_test=test_y, actuals_and_forecast_df=test_df,targets=test_df.iloc[:,0:16].columns.values.tolist())
    
            results = results.append(actuals_and_forecast_df)
            start_time = start_time + td(hours=8)
            
        results.to_csv(path  + ".csv", index = False)
        
          
        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()
    
    
    


def load_data_BM(file_path):
    # Define date format and parsing function
    date_format = "%m/%d/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    
    # Read the CSV file
    dat = pd.read_csv(file_path, index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)
    
    # Drop unnecessary columns and handle missing values
    dat = dat.drop(["index"], axis=1)
    dat = dat.bfill(axis='rows')
    dat = dat.ffill(axis='rows')
    dat = dat._get_numeric_data()
    
    # Select target columns
    Y = dat.iloc[:, 0:16]
    
    return dat, Y


def generate_train_and_test_dataframes_LGBM_DAM(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                        test_start_time: dt, test_end_time: dt):

    train_X = None
    train_y = None 
    test_X = None 
    test_y = None 
    test_df = None

    try:
        
        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
            return train_X, train_y, test_X, test_y, test_df
        
        original_columns = list(participant_df.columns)

        participant_df = participant_df.dropna()

        date_format = "%d/%m/%Y %H:%M"

        train_df = None
        

        
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)    
        train_df = participant_df[(participant_df.index>=train_start_time_str) & (participant_df.index<train_end_time_str)].copy(deep="True")

        if train_df is None or len(train_df) == 0:
            print("Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df


        
        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)    
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format) 
        test_df = participant_df[(participant_df.index>=test_start_time_str) & (participant_df.index<test_end_time_str)].copy(deep="True")

        if test_df is None or len(test_df) == 0:
            print("Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df


        train_X = train_df.iloc[:, 24:]
        test_X = test_df.iloc[:, 24:]
        train_y = train_df.iloc[:, 0:24]
        test_y = test_df.iloc[:, 0:24]
                                
        return train_X, train_y, test_X, test_y, test_df
        
    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df
    
    
def fit_multitarget_model_LGBM_DAM(model_1, model_2, model_3, model_4, model_5, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):

    try:
        
        model_1.fit(X_train, Y_train) 
        model_2.fit(X_train, Y_train) 
        model_3.fit(X_train, Y_train) 
        model_4.fit(X_train, Y_train) 
        model_5.fit(X_train, Y_train) 


        model_test_predictions_1=None  
        model_test_predictions_3=None  
        model_test_predictions_5=None  
        model_test_predictions_7=None  
        model_test_predictions_9=None  
        model_test_predictions_1 = model_1.predict(X_test)     
        model_test_predictions_3 = model_2.predict(X_test) 
        model_test_predictions_5 = model_3.predict(X_test)     
        model_test_predictions_7 = model_4.predict(X_test)     
        model_test_predictions_9 = model_5.predict(X_test)          
                    
        cols = Y_test.iloc[:, 0:16].columns.values.tolist()   

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_10"] = model_test_predictions_1[:,i].tolist() if len(cols) > 1 else model_test_predictions_1.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_30"] = model_test_predictions_3[:,i].tolist() if len(cols) > 1 else model_test_predictions_3.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_50"] = model_test_predictions_5[:,i].tolist() if len(cols) > 1 else model_test_predictions_5.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_70"] = model_test_predictions_7[:,i].tolist() if len(cols) > 1 else model_test_predictions_7.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_90"] = model_test_predictions_9[:,i].tolist() if len(cols) > 1 else model_test_predictions_9.tolist() 
          
           
        return actuals_and_forecast_df
    
    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()
    
def rolling_walk_forward_validation_LGBM_DAM(model_1, model_2, model_3, model_4, model_5, data, targets, start_time, end_time, training_days, path):

    try:

        all_columns = list(data.columns)            
        results = pd.DataFrame()
            

        date_format = "%d/%m/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)
        
        while start_time < end_time:
            
            #Train interval
            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time 
    
            #Test interval, the test period is always the day ahead forecast
            test_start_time = train_end_time + td(hours=24)
            test_end_time = test_start_time + td(hours=1)
            
            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))
    
            #Generate the calibration and test dataframes.
            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes_LGBM_DAM(participant_df=data, train_start_time=train_start_time.strftime("%d/%m/%Y %H:%M"), train_end_time=train_end_time.strftime("%d/%m/%Y %H:%M"), 
                            test_start_time=test_start_time.strftime("%d/%m/%Y %H:%M"), test_end_time=test_end_time.strftime("%d/%m/%Y %H:%M"))
            
            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
    
            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
            
            actuals_and_forecast_df = fit_multitarget_model_LGBM_DAM(model_1=model_1,model_2=model_2,model_3=model_3, model_4=model_4,model_5=model_5, X_train=train_X, Y_train=train_y,
                                            X_test=test_X, Y_test=test_y, actuals_and_forecast_df=test_df,targets=test_df.iloc[:,0:16].columns.values.tolist())
    
            results = pd.concat([results, actuals_and_forecast_df])

            start_time = start_time + td(hours=24)
            
        results.to_csv(path  + ".csv", index = False)
        
          
        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()    
    
    
    
def generate_train_and_test_dataframes_LGBM_BM(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                        test_start_time: dt, test_end_time: dt):

    train_X = None
    train_y = None 
    test_X = None 
    test_y = None 
    test_df = None

    try:
        
        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
            return train_X, train_y, test_X, test_y, test_df
        
        original_columns = list(participant_df.columns)

        participant_df = participant_df.dropna()

        date_format="%m/%d/%Y %H:%M"

        train_df = None
        

        
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)    
        train_df = participant_df[(participant_df.index>=train_start_time_str) & (participant_df.index<train_end_time_str)].copy(deep="True")

        if train_df is None or len(train_df) == 0:
            print("Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df


        
        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)    
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format) 
        test_df = participant_df[(participant_df.index>=test_start_time_str) & (participant_df.index<test_end_time_str)].copy(deep="True")

        if test_df is None or len(test_df) == 0:
            print("Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")            
            return train_X, train_y, test_X, test_y, test_df


        train_X = train_df.iloc[:, 16:]
        test_X = test_df.iloc[:, 16:]
        train_y = train_df.iloc[:, 0:16]
        test_y = test_df.iloc[:, 0:16]
                                
        return train_X, train_y, test_X, test_y, test_df
        
    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df
    
    
def fit_multitarget_model_LGBM_BM(model_1, model_2, model_3, model_4, model_5, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):

    try:
        
        model_1.fit(X_train, Y_train) 
        model_2.fit(X_train, Y_train) 
        model_3.fit(X_train, Y_train) 
        model_4.fit(X_train, Y_train) 
        model_5.fit(X_train, Y_train) 

        model_test_predictions_1=None  
        model_test_predictions_3=None  
        model_test_predictions_5=None  
        model_test_predictions_7=None  
        model_test_predictions_9=None 
        
        model_test_predictions_1 = model_1.predict(X_test)     
        model_test_predictions_3 = model_2.predict(X_test) 
        model_test_predictions_5 = model_3.predict(X_test)     
        model_test_predictions_7 = model_4.predict(X_test)     
        model_test_predictions_9 = model_5.predict(X_test)          
                    
        cols = Y_test.iloc[:, 0:16].columns.values.tolist()   

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_10"] = model_test_predictions_1[:,i].tolist() if len(cols) > 1 else model_test_predictions_1.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_30"] = model_test_predictions_3[:,i].tolist() if len(cols) > 1 else model_test_predictions_3.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_50"] = model_test_predictions_5[:,i].tolist() if len(cols) > 1 else model_test_predictions_5.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_70"] = model_test_predictions_7[:,i].tolist() if len(cols) > 1 else model_test_predictions_7.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_90"] = model_test_predictions_9[:,i].tolist() if len(cols) > 1 else model_test_predictions_9.tolist() 
          
           
        return actuals_and_forecast_df
    
    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()
    
def rolling_walk_forward_validation_LGBM_BM(model_1, model_2, model_3, model_4, model_5, data, targets, start_time, end_time, training_days, path):

    try:

        all_columns = list(data.columns)            
        results = pd.DataFrame()
            
        #Each time we 
        # (a) fit the model on the calibration/train data
        # (b) apply it to the test data i.e. forecast 1 day ahead.
        #Repeat.
        date_format="%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)
        
        while start_time < end_time:
            
            #Train interval
            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time 
    
            #Test interval, the test period is always the day ahead forecast
            test_start_time = train_end_time + td(hours=8)
            test_end_time = test_start_time + td(minutes=30)
            
            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))
    
            #Generate the calibration and test dataframes.
            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes_LGBM_BM(participant_df=data, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"), train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"), 
                            test_start_time=test_start_time.strftime("%m/%d/%Y %H:%M"), test_end_time=test_end_time.strftime("%m/%d/%Y %H:%M"))
            
            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
    
            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
            
            actuals_and_forecast_df = fit_multitarget_model_LGBM_BM(model_1=model_1,model_2=model_2,model_3=model_3, model_4=model_4,model_5=model_5, X_train=train_X, Y_train=train_y,
                                            X_test=test_X, Y_test=test_y, actuals_and_forecast_df=test_df,targets=test_df.iloc[:,0:16].columns.values.tolist())
    
            results = pd.concat([results, actuals_and_forecast_df])
            start_time = start_time + td(hours=8)
            
        results.to_csv(path  + ".csv", index = False)
        
          
        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()    
    
    

