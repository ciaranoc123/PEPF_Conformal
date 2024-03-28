import os;
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from datetime import timedelta as td
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt
from functools import reduce
import importlib
import datetime as dt
from datetime import datetime
from math import floor
import seaborn as sns
from math import sqrt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn import preprocessing as prep
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, mean_absolute_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import warnings
    
    
    
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LassoLarsIC
import datetime as dt

def load_and_preprocess_data_LEAR_DAM(csv_file):
    # Define date format and parser
    date_format = "%d/%m/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    
    # Load the data
    dat = pd.read_csv(csv_file, index_col="DeliveryPeriod", parse_dates=True, date_parser=date_parse)
    dat = pd.DataFrame(dat)
    
    # Fill missing values
    dat = dat.bfill(axis='rows')
    dat = dat.ffill(axis='rows')

    # Split into features and target
    Y = dat.iloc[:, 0:24]
    X = dat.iloc[:, 24:]
    
    # Split into train and test sets
    X_train = X.iloc[:12360, :]
    Y_train = Y.iloc[:12360, :]
    X_test = X.iloc[12360:13104, :]
    Y_test = Y.iloc[12360:13104, :]

    # Split features further for different components
    rnn_train_1 = X_train.loc[:, "EURPrices-24":"EURPrices-167"]
    rnn_train_2 = X_train.loc[:, "WF": "WF-143"]
    rnn_train_3 = X_train.loc[:, "DF": "DF-143"]

    rnn_test_1 = X_test.loc[:, "EURPrices-24":"EURPrices-167"]
    rnn_test_2 = X_test.loc[:, "WF": "WF-143"]
    rnn_test_3 = X_test.loc[:, "DF": "DF-143"]

    # Split the target variable
    rnn_Y = Y_train.loc[:, "EURPrices":"EURPrices+23"]
    rnn_test_Y = Y_test.loc[:, "EURPrices":"EURPrices+23"]

    # Scale the features
    X_scaler1 = preprocessing.MinMaxScaler()
    X_scaler2 = preprocessing.MinMaxScaler()
    X_scaler3 = preprocessing.MinMaxScaler()
    Y_scaler = preprocessing.MinMaxScaler()

    rnn_scaled_train_1 = X_scaler1.fit_transform(rnn_train_1)
    rnn_scaled_train_2 = X_scaler2.fit_transform(rnn_train_2)
    rnn_scaled_train_3 = X_scaler3.fit_transform(rnn_train_3)

    # Scale the target variable
    Y_train_Scaled = Y_scaler.fit_transform(rnn_Y)
    Y_test_scaled = Y_scaler.transform(rnn_test_Y)

    # Concatenate the scaled features
    X_train_Scaled = np.concatenate((rnn_scaled_train_1, rnn_scaled_train_2, rnn_scaled_train_3), axis=1)

    # Calculate alpha using LassoLarsIC
    alpha = LassoLarsIC(criterion='aic', max_iter=2500).fit(X_train_Scaled, Y_train_Scaled[:,:1].ravel()).alpha_
    
    return dat, Y, alpha
    
    
    
def generate_train_and_test_dataframes_LEAR_DAM(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):


    train_X = None
    train_y = None
    test_X = None
    test_y = None
    test_df = None
    train_df = None

    try:

        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
        #             return train_X, train_y, test_X, test_y, test_df

        original_columns = list(participant_df.columns)

        # Remove any rows with nan's etc (there shouldn't be any in the input).
        participant_df = participant_df.dropna()

        date_format = "%d/%m/%Y %H:%M"

        # The train dataframe, it will be used later to create train_X and train_y.
        train_df = None
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[
            (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
            deep="True")

        if train_df is None or len(train_df) == 0:
            print(
                "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")
        #             return train_X, train_y, test_X, test_y, test_df

        # Create the test dataframe, it will be used later to create test_X and test_y
        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
        test_df = participant_df[
            (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
            deep="True")

        if test_df is None or len(test_df) == 0:
            print(
                "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")
        #             return train_X, train_y, test_X, test_y, test_df

        rnn_train_1 = train_df.loc[:, "EURPrices-24":"EURPrices-167"]
        rnn_train_2 = train_df.loc[:, "WF": "WF-143"]
        rnn_train_3 = train_df.loc[:, "DF": "DF-143"]

        rnn_test_1 = test_df.loc[:, "EURPrices-24":"EURPrices-167"]
        rnn_test_2 = test_df.loc[:, "WF": "WF-143"]
        rnn_test_3 = test_df.loc[:, "DF": "DF-143"]

        rnn_Y = train_df.loc[:, "EURPrices":"EURPrices+23"]

        X_scaler1 = preprocessing.MinMaxScaler()
        X_scaler2 = preprocessing.MinMaxScaler()
        X_scaler3 = preprocessing.MinMaxScaler()
        Y_scaler = preprocessing.MinMaxScaler()

        rnn_scaled_train_1 = X_scaler1.fit_transform(rnn_train_1)
        rnn_scaled_train_2 = X_scaler2.fit_transform(rnn_train_2)
        rnn_scaled_train_3 = X_scaler3.fit_transform(rnn_train_3)

        train_y = Y_scaler.fit_transform(rnn_Y)
        Y_scaler_n = Y_scaler.fit(rnn_Y)

        train_X=np.concatenate((rnn_scaled_train_1, rnn_scaled_train_2, rnn_scaled_train_3), axis=1)
        test_X=np.concatenate((X_scaler1.transform(rnn_test_1), X_scaler2.transform(rnn_test_2), X_scaler3.transform(rnn_test_3)), axis=1)

        test_y = test_df.iloc[:, 0:24]

        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n



    
def fit_multitarget_model_LEAR_DAM(model_1, model_2, model_3, model_4, model_5, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets, Y_scaler_n):
    try:
        Y_scaler = preprocessing.MinMaxScaler()
        Y_scaler = Y_scaler.fit(Y_train)
        cols = Y_test.iloc[:, 0:24].columns.values.tolist()

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
        model_test_predictions_1 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_1.predict(X_test)).reshape(1,24)), columns=cols)    
        model_test_predictions_3 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_2.predict(X_test)).reshape(1,24)), columns=cols)  
        model_test_predictions_5 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_3.predict(X_test)).reshape(1,24)), columns=cols)     
        model_test_predictions_7 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_4.predict(X_test)).reshape(1,24)), columns=cols)     
        model_test_predictions_9 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_5.predict(X_test)).reshape(1,24)), columns=cols) 


        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_10"] = model_test_predictions_1.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_1.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_30"] = model_test_predictions_3.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_3.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_50"] = model_test_predictions_5.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_5.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_70"] = model_test_predictions_7.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_7.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_90"] = model_test_predictions_9.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_9.tolist() 

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


def rolling_walk_forward_validation_LEAR_DAM(model_1, model_2, model_3, model_4, model_5, data, targets, start_time, end_time, training_days, path):
    try:

        all_columns = list(data.columns)
        results = pd.DataFrame()

        date_format = "%d/%m/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)

        while start_time < end_time:

            # Train interval
            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time

            # Test interval, the test period is always the day ahead forecast
            test_start_time = train_end_time + td(hours=24)
            test_end_time = test_start_time + td(hours=1)

            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            # Generate the calibration and test dataframes.
            train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n = generate_train_and_test_dataframes_LEAR_DAM(
                participant_df=data, train_start_time=train_start_time.strftime("%d/%m/%Y %H:%M"),
                train_end_time=train_end_time.strftime("%d/%m/%Y %H:%M"),
                test_start_time=test_start_time.strftime("%d/%m/%Y %H:%M"),
                test_end_time=test_end_time.strftime("%d/%m/%Y %H:%M"))

            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(
                    train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(
                    test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue

            # Fit the model to the train datasets, produce a forecast and return a dataframe containing the forecast/actuals.
            actuals_and_forecast_df = fit_multitarget_model_LEAR_DAM(model_1=model_1,model_2=model_2,model_3=model_3, model_4=model_4,model_5=model_5, 
                                                            Y_scaler_n=Y_scaler_n, X_train=train_X, Y_train=train_y, X_test=test_X, Y_test=test_y,
                                                            actuals_and_forecast_df=test_df.iloc[:, 0:24], targets=test_df.iloc[:,0:24].columns.values.tolist())

            results = pd.concat([results, actuals_and_forecast_df])
            start_time = start_time + td(hours=24)

        results.to_csv(path + ".csv", index=False)



    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()    
    


def load_and_preprocess_data_LEAR_BM(file_path):
    date_format = "%m/%d/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    
    # Read the CSV file
    dat = pd.read_csv(file_path, index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)
    
    # Drop unnecessary columns and handle missing values
    dat = dat.drop(["index"], axis=1)
    dat = dat.bfill(axis='rows')
    dat = dat.ffill(axis='rows')
    dat = dat._get_numeric_data()
    
    # Splitting data into features and target
    Y = dat.iloc[:, 0:16]
    X = dat.iloc[:, 16:]
    X_train = X.iloc[:7250, :]
    Y_train = Y.iloc[:7250, :]
    X_test = X.iloc[7250:8739, :]
    Y_test = Y.iloc[7250:8739, :]
    
    # Selecting relevant columns for RNN training
    rnn_train_1 = X_train.loc[:, "lag_-3x1": "lag_-50x1"]
    rnn_train_2 = X_train.loc[:, "lag_-3x2": "lag_-50x2"]
    rnn_train_3 = X_train.loc[:, "lag_-2x3":"lag_-49x3"]
    rnn_train_4 = X_train.loc[:, "lag_0x6": "lag_-47x6"]
    rnn_train_5 = X_train.loc[:, "lag_-2x12": "lag_-49x12"]
    rnn_train_6 = X_train.loc[:, "lag_2x7": "lag_17x7"]
    rnn_train_7 = X_train.loc[:, "lag_2x8": "lag_17x8"]
    rnn_train_8 = X_train.loc[:, "lag_2x9": "lag_17x9"]
    rnn_train_9 = X_train.loc[:, "lag_2x10":"lag_17x10"]
    rnn_train_10 = X_train.loc[:, "lag_2x11":"lag_17x11"]
    
    rnn_Y = Y_train.loc[:, "lag_2y":"lag_17y"]
    
    # Scaling features and target
    X_scalers = [preprocessing.MinMaxScaler() for _ in range(10)]
    Y_scaler = preprocessing.MinMaxScaler()
    
    rnn_scaled_train = []
    for i, rnn_train in enumerate([rnn_train_1, rnn_train_2, rnn_train_3, rnn_train_4, rnn_train_5,
                                    rnn_train_6, rnn_train_7, rnn_train_8, rnn_train_9, rnn_train_10]):
        rnn_scaled_train.append(X_scalers[i].fit_transform(rnn_train))
    
    Y_train_scaled = Y_scaler.fit_transform(Y_train)
    
    X_train_scaled = np.concatenate(rnn_scaled_train, axis=1)
    
    # Estimating alpha using LassoLarsIC
    alpha = LassoLarsIC(criterion='aic', max_iter=2500).fit(X_train_scaled, Y_train_scaled[:, :1].ravel()).alpha_
    
    return dat, alpha, Y



    
    
def generate_train_and_test_dataframes_LEAR_BM(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):


    # These are the dataframes that will be returned from the method.
    train_X = None
    train_y = None
    test_X = None
    test_y = None
    test_df = None
    train_df = None

    try:
        
        if len(participant_df) == 0:
            print("Warning: generate_train_and_test_dataframes method, participant_df has 0 rows. Ending.")
#             return train_X, train_y, test_X, test_y, test_df
        
        original_columns = list(participant_df.columns)

        #Remove any rows with nan's etc (there shouldn't be any in the input).        
        participant_df = participant_df.dropna()
        
        date_format="%m/%d/%Y %H:%M"
      
        train_df = None
        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[(participant_df.index>=train_start_time_str) & (participant_df.index<train_end_time_str)].copy(deep="True")

        if train_df is None or len(train_df) == 0:
            print("Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")            

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)    
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format) 
        test_df = participant_df[(participant_df.index>=test_start_time_str) & (participant_df.index<test_end_time_str)].copy(deep="True")

        if test_df is None or len(test_df) == 0:
            print("Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")            


        rnn_train_1 = train_df.loc[:,"lag_-3x1": "lag_-50x1"]
        rnn_train_2 = train_df.loc[:,"lag_-3x2":"lag_-50x2"]
        rnn_train_3 = train_df.loc[:,"lag_-2x3": "lag_-49x3"]
        rnn_train_4 = train_df.loc[:,"lag_0x6": "lag_-47x6"]
        rnn_train_5 = train_df.loc[:,"lag_-2x12": "lag_-49x12"]
        rnn_train_6 = train_df.loc[:,"lag_2x7":"lag_17x7"]
        rnn_train_7 = train_df.loc[:,"lag_2x8": "lag_17x8"]
        rnn_train_8 = train_df.loc[:,"lag_2x9":"lag_17x9"]
        rnn_train_9 = train_df.loc[:,"lag_2x10":"lag_17x10"]
        rnn_train_10 = train_df.loc[:,"lag_2x11": "lag_17x11"]

        rnn_test_1 = test_df.loc[:,"lag_-3x1": "lag_-50x1"]
        rnn_test_2 = test_df.loc[:,"lag_-3x2":"lag_-50x2"]
        rnn_test_3 = test_df.loc[:,"lag_-2x3":"lag_-49x3"]
        rnn_test_4 = test_df.loc[:,"lag_0x6":"lag_-47x6"]
        rnn_test_5 = test_df.loc[:,"lag_-2x12":"lag_-49x12"]
        rnn_test_6 = test_df.loc[:,"lag_2x7":"lag_17x7"]
        rnn_test_7 = test_df.loc[:,"lag_2x8":"lag_17x8"]
        rnn_test_8 = test_df.loc[:,"lag_2x9":"lag_17x9"]
        rnn_test_9 = test_df.loc[:,"lag_2x10":"lag_17x10"]
        rnn_test_10 = test_df.loc[:,"lag_2x11":"lag_17x11"]

        rnn_Y = train_df.loc[:,"lag_2y":"lag_17y"]

        X_scaler_1 = preprocessing.MinMaxScaler()
        X_scaler_2 = preprocessing.MinMaxScaler()
        X_scaler_3 = preprocessing.MinMaxScaler()
        X_scaler_4 = preprocessing.MinMaxScaler()
        X_scaler_5 = preprocessing.MinMaxScaler()
        X_scaler_6 = preprocessing.MinMaxScaler()
        X_scaler_7 = preprocessing.MinMaxScaler()
        X_scaler_8 = preprocessing.MinMaxScaler()
        X_scaler_9 = preprocessing.MinMaxScaler()
        X_scaler_10 = preprocessing.MinMaxScaler()

        Y_scaler = preprocessing.MinMaxScaler()

        rnn_scaled_train_1 = X_scaler_1.fit_transform(rnn_train_1)
        rnn_scaled_train_2 = X_scaler_2.fit_transform(rnn_train_2)
        rnn_scaled_train_3 = X_scaler_3.fit_transform(rnn_train_3)
        rnn_scaled_train_4 = X_scaler_4.fit_transform(rnn_train_4)
        rnn_scaled_train_5 = X_scaler_5.fit_transform(rnn_train_5)
        rnn_scaled_train_6 = X_scaler_6.fit_transform(rnn_train_6)
        rnn_scaled_train_7 = X_scaler_7.fit_transform(rnn_train_7)
        rnn_scaled_train_8 = X_scaler_8.fit_transform(rnn_train_8)
        rnn_scaled_train_9 = X_scaler_9.fit_transform(rnn_train_9)
        rnn_scaled_train_10 = X_scaler_10.fit_transform(rnn_train_10)

        train_y = Y_scaler.fit_transform(rnn_Y)
        Y_scaler_n=Y_scaler.fit(rnn_Y)
        
        train_X=np.concatenate((rnn_scaled_train_1, rnn_scaled_train_2, rnn_scaled_train_3, rnn_scaled_train_4, rnn_scaled_train_5,
                                rnn_scaled_train_6, rnn_scaled_train_7, rnn_scaled_train_8, rnn_scaled_train_9, rnn_scaled_train_10), axis=1)
        test_X=np.concatenate((X_scaler_1.transform(rnn_test_1), X_scaler_2.transform(rnn_test_2), X_scaler_3.transform(rnn_test_3), X_scaler_4.transform(rnn_test_4), X_scaler_5.transform(rnn_test_5), 
                               X_scaler_6.transform(rnn_test_6), X_scaler_7.transform(rnn_test_7), X_scaler_8.transform(rnn_test_8),X_scaler_9.transform(rnn_test_9), X_scaler_10.transform(rnn_test_10)), axis=1)


        test_y = test_df.iloc[:, 0:16]

        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n


def fit_multitarget_model_LEAR_BM(model_1, model_2, model_3, model_4, model_5, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets, Y_scaler_n):
    try:
        Y_scaler = preprocessing.MinMaxScaler()
        Y_scaler = Y_scaler.fit(Y_train)
        cols = Y_test.iloc[:, 0:16].columns.values.tolist()

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
        model_test_predictions_1 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_1.predict(X_test)).reshape(1,16)), columns=cols)    
        model_test_predictions_3 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_2.predict(X_test)).reshape(1,16)), columns=cols)  
        model_test_predictions_5 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_3.predict(X_test)).reshape(1,16)), columns=cols)     
        model_test_predictions_7 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_4.predict(X_test)).reshape(1,16)), columns=cols)     
        model_test_predictions_9 = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model_5.predict(X_test)).reshape(1,16)), columns=cols)
        


        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_10"] = model_test_predictions_1.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_1.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_30"] = model_test_predictions_3.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_3.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_50"] = model_test_predictions_5.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_5.tolist() 
            
        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_70"] = model_test_predictions_7.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_7.tolist() 

        for i in range(0,len(cols)):    
            actuals_and_forecast_df[cols[i]+"_Forecast_90"] = model_test_predictions_9.iloc[:,i].T.tolist() if len(cols) > 1 else model_test_predictions_9.tolist() 

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


  
    
def rolling_walk_forward_validation_LEAR_BM(model_1, model_2, model_3, model_4, model_5, data, targets, start_time, end_time, training_days, path):
 
    try:

        all_columns = list(data.columns)            
        results = pd.DataFrame()
            

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
            train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n= generate_train_and_test_dataframes_LEAR_BM(
                participant_df=data, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"), train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"), 
                test_start_time=test_start_time.strftime("%m/%d/%Y %H:%M"), test_end_time=test_end_time.strftime("%m/%d/%Y %H:%M"))
            
            
            if train_X is None or len(train_X) == 0:
                print("Don't have a train dataframe for train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
    
            if test_X is None or len(test_X) == 0:
                print("Don't have a test dataframe for test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time) + ", skipping.")
                start_time = start_time + td(days=training_days)
                continue
            
            #Fit the model to the train datasets, produce a forecast and return a dataframe containing the forecast/actuals.
            actuals_and_forecast_df = fit_multitarget_model_LEAR_BM(model_1=model_1,model_2=model_2,model_3=model_3, model_4=model_4,model_5=model_5,
                                                            Y_scaler_n=Y_scaler_n, X_train=train_X, Y_train=train_y, X_test=test_X, Y_test=test_y,
                                                            actuals_and_forecast_df=test_df.iloc[:,0:16], targets=test_df.iloc[:,0:16].columns.values.tolist())

    
            results = results.append(actuals_and_forecast_df)
        
            start_time = start_time + td(hours=8)
            
        results.to_csv(path  + ".csv", index = False)
        

        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()
    
    
    
    
    
    
    
    
    
    
    
    
    
    

