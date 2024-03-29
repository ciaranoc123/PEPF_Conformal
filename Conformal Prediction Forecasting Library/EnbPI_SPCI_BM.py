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
from sklearn_quantile import RandomForestQuantileRegressor, KNeighborsQuantileRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import SPCI_class as SPCI
import torch
from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn import preprocessing

import warnings


   


    
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
    Y = dat.iloc[:, 0:1]
    Y1 = dat.iloc[:, 0:16]
        
    return dat, Y, Y1    
    
    
    
def generate_train_and_test_dataframes_BM(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
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

        date_format = "%m/%d/%Y %H:%M"

        train_df = None

        train_start_time_str = dt.datetime.strptime(train_start_time, date_format)
        train_end_time_str = dt.datetime.strptime(train_end_time, date_format)
        train_df = participant_df[
            (participant_df.index >= train_start_time_str) & (participant_df.index < train_end_time_str)].copy(
            deep="True")

        if train_df is None or len(train_df) == 0:
            print(
                "Don't have a train dataframe for train_start_time: " + train_start_time_str + ", train_end_time: " + train_end_time_str + ", exiting.")
            return train_X, train_y, test_X, test_y, test_df

        test_start_time_str = dt.datetime.strptime(test_start_time, date_format)
        test_end_time_str = dt.datetime.strptime(test_end_time, date_format)
        test_df = participant_df[
            (participant_df.index >= test_start_time_str) & (participant_df.index < test_end_time_str)].copy(
            deep="True")

        if test_df is None or len(test_df) == 0:
            print(
                "Don't have a test dataframe for test_start_time: " + test_start_time_str + ", test_end_time: " + test_end_time_str + ", exiting.")
            return train_X, train_y, test_X, test_y, test_df
        
        x_1a = train_df.loc[:, "lag_-18x1": "lag_-50x1"]
        x_1b = train_df.loc[:, "lag_-18x2": "lag_-50x2"]
        x_1c = train_df.loc[:, "lag_-17x3":"lag_-49x3"]
        x_1d = train_df.loc[:, "lag_-16x6": "lag_-47x6"]
        x_1e = train_df.loc[:, "lag_-17x12": "lag_-49x12"]
        x_1f = train_df.loc[:, "lag_2x7": "lag_17x7"]
        x_1g = train_df.loc[:, "lag_2x8": "lag_17x8"]
        x_1h = train_df.loc[:, "lag_2x9": "lag_17x9"]
        x_1i = train_df.loc[:, "lag_2x10":"lag_17x10"]
        x_1j = train_df.loc[:, "lag_2x11":"lag_17x11"]
        X_train_d = pd.concat([x_1a, x_1b, x_1c, x_1d,x_1e,x_1f, x_1g, x_1h, x_1i, x_1j], axis=1)   

        x_2a = test_df.loc[:, "lag_-18x1": "lag_-50x1"]  # Fixed this line
        x_2b = test_df.loc[:, "lag_-18x2": "lag_-50x2"]  # Fixed this line
        x_2c = test_df.loc[:, "lag_-17x3":"lag_-49x3"]    # Fixed this line
        x_2d = test_df.loc[:, "lag_-16x6": "lag_-47x6"]   # Fixed this line
        x_2e = test_df.loc[:, "lag_-17x12": "lag_-49x12"] # Fixed this line
        x_2f = test_df.loc[:, "lag_2x7": "lag_17x7"]      # Fixed this line
        x_2g = test_df.loc[:, "lag_2x8": "lag_17x8"]      # Fixed this line
        x_2h = test_df.loc[:, "lag_2x9": "lag_17x9"]      # Fixed this line
        x_2i = test_df.loc[:, "lag_2x10":"lag_17x10"]     # Fixed this line
        x_2j = test_df.loc[:, "lag_2x11":"lag_17x11"]     # Fixed this line
        X_test_d = pd.concat([x_2a, x_2b, x_2c, x_2d,x_2e,x_2f, x_2g, x_2h, x_2i, x_2j], axis=1)  # Fixed
        
        train_X = X_train_d
        test_X = X_test_d
        train_y = train_df.iloc[:, 0:1]
        test_y = test_df.iloc[:, 0:1]

        return train_X, train_y, test_X, test_y, test_df

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df


def fit_multitarget_model_EnbPI_BM(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):
    try:
        cols = Y_train.columns.values.tolist()
        model.fit(X_train, Y_train)

        X_train_t1 = torch.from_numpy(np.array(X_train))
        X_test_t1 = torch.from_numpy(np.array(X_test))
        Y_train_t1 = torch.from_numpy(np.array(Y_train))
        Y_test_t1 = torch.from_numpy(np.array(Y_test))

        X_train_t2 = torch.from_numpy(np.array(X_train))
        X_test_t2 = torch.from_numpy(np.array(X_test))
        Y_train_t2 = torch.from_numpy(np.array(Y_train))
        Y_test_t2 = torch.from_numpy(np.array(Y_test))

        fit_func = model
        stride = 16
        SPCI_class_1 = SPCI.SPCI_and_EnbPI(X_train_t1, X_test_t1, Y_train_t1, Y_test_t1, fit_func=fit_func)
        SPCI_class_1.fit_bootstrap_models_online_multistep(B=20, fit_sigmaX=False, stride=stride)
        use_SPCI = False
        smallT = not use_SPCI
        SPCI_class_1.compute_PIs_Ensemble_online(0.1, smallT=smallT, past_window=365, use_SPCI=use_SPCI,
                                                 quantile_regr=False, stride=stride)
        Pred_ints_1 = SPCI_class_1.PIs_Ensemble
        model_test_predictions_1 = np.array(Pred_ints_1)
        model_test_predictions_1 = model_test_predictions_1.reshape(16, 2)

        SPCI_class_2 = SPCI.SPCI_and_EnbPI(X_train_t2, X_test_t2, Y_train_t2, Y_test_t2, fit_func=fit_func)
        SPCI_class_2.fit_bootstrap_models_online_multistep(B=20, fit_sigmaX=False, stride=stride)
        use_SPCI = False
        smallT = not use_SPCI
        SPCI_class_2.compute_PIs_Ensemble_online(0.3, smallT=smallT, past_window=365, use_SPCI=use_SPCI,
                                                 quantile_regr=False, stride=stride)
        Pred_ints_2 = SPCI_class_2.PIs_Ensemble
        model_test_predictions_2 = np.array(Pred_ints_2)
        model_test_predictions_2 = model_test_predictions_2.reshape(16, 2)

        print(model_test_predictions_1)

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i]] = np.array(Y_test_t1[i, :]).tolist() if len(
                cols) > 1 else np.array(Y_test_t1).tolist()


        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_10"] = model_test_predictions_1[i, 0:1].tolist() if len(
                cols) > 1 else model_test_predictions_1.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_30"] = model_test_predictions_2[i, 0:1].tolist() if len(
                cols) > 1 else model_test_predictions_2.tolist()


        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_70"] = model_test_predictions_2[i, 1:2].tolist() if len(
                cols) > 1 else model_test_predictions_2.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_90"] = model_test_predictions_1[i, 1:2].tolist() if len(
                cols) > 1 else model_test_predictions_1.tolist()

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


def rolling_walk_forward_validation_EnbPI_BM(model, data, targets, start_time, end_time, training_days, path):
    try:

        all_columns = list(data.columns)
        results = pd.DataFrame()

        date_format = "%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)

        while start_time < end_time:

            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time

            test_start_time = train_end_time + td(hours=0)
            test_end_time = test_start_time + td(hours=8)

            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes_BM(participant_df=data,
                                                                                           train_start_time=train_start_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"),
                                                                                           train_end_time=train_end_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"),
                                                                                           test_start_time=test_start_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"),
                                                                                           test_end_time=test_end_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"))

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

            actuals_and_forecast_df = fit_multitarget_model_EnbPI_BM(model=model, X_train=train_X, Y_train=train_y,
                                                            X_test=test_X, Y_test=test_y,
                                                            actuals_and_forecast_df=pd.DataFrame(),
                                                            targets=test_df.iloc[:,0:16].columns.values.tolist())

            results = pd.concat([results, actuals_and_forecast_df])
            start_time = start_time + td(hours=8)

        results.to_csv(path + ".csv", index=False)



    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()

    
    
def fit_multitarget_model_SPCI_BM(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):
    try:
        cols = Y_train.columns.values.tolist()
        model.fit(X_train, Y_train)

        X_train_t1 = torch.from_numpy(np.array(X_train))
        X_test_t1 = torch.from_numpy(np.array(X_test))
        Y_train_t1 = torch.from_numpy(np.array(Y_train))
        Y_test_t1 = torch.from_numpy(np.array(Y_test))

        X_train_t2 = torch.from_numpy(np.array(X_train))
        X_test_t2 = torch.from_numpy(np.array(X_test))
        Y_train_t2 = torch.from_numpy(np.array(Y_train))
        Y_test_t2 = torch.from_numpy(np.array(Y_test))

        fit_func = model
        stride = 16
        SPCI_class_1 = SPCI.SPCI_and_EnbPI(X_train_t1, X_test_t1, Y_train_t1, Y_test_t1, fit_func=fit_func)
        SPCI_class_1.fit_bootstrap_models_online_multistep(B=10, fit_sigmaX=False, stride=stride)
        use_SPCI = True
        smallT = not use_SPCI
        SPCI_class_1.compute_PIs_Ensemble_online(0.1, smallT=smallT, past_window=1600, use_SPCI=use_SPCI,
                                                 quantile_regr=True, stride=stride)
        Pred_ints_1 = SPCI_class_1.PIs_Ensemble
        model_test_predictions_1 = np.array(Pred_ints_1)
        model_test_predictions_1 = model_test_predictions_1.reshape(16, 2)

        SPCI_class_2 = SPCI.SPCI_and_EnbPI(X_train_t2, X_test_t2, Y_train_t2, Y_test_t2, fit_func=fit_func)
        SPCI_class_2.fit_bootstrap_models_online_multistep(B=10, fit_sigmaX=False, stride=stride)
        use_SPCI = True
        smallT = not use_SPCI
        SPCI_class_2.compute_PIs_Ensemble_online(0.3, smallT=smallT, past_window=1600, use_SPCI=use_SPCI,
                                                 quantile_regr=True, stride=stride)
        Pred_ints_2 = SPCI_class_2.PIs_Ensemble
        model_test_predictions_2 = np.array(Pred_ints_2)
        model_test_predictions_2 = model_test_predictions_2.reshape(16, 2)

        print(model_test_predictions_1)

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i]] = np.array(Y_test_t1[i, :]).tolist() if len(
                cols) > 1 else np.array(Y_test_t1).tolist()


        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_10"] = model_test_predictions_1[i, 0:1].tolist() if len(
                cols) > 1 else model_test_predictions_1.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_30"] = model_test_predictions_2[i, 0:1].tolist() if len(
                cols) > 1 else model_test_predictions_2.tolist()


        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_70"] = model_test_predictions_2[i, 1:2].tolist() if len(
                cols) > 1 else model_test_predictions_2.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_90"] = model_test_predictions_1[i, 1:2].tolist() if len(
                cols) > 1 else model_test_predictions_1.tolist()

        print("train number of observations: " + str(len(Y_train)))

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()    
    
def rolling_walk_forward_validation_SPCI_BM(model, data, targets, start_time, end_time, training_days, path):
    try:

        all_columns = list(data.columns)
        results = pd.DataFrame()

        date_format = "%m/%d/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)

        while start_time < end_time:

            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time

            test_start_time = train_end_time + td(hours=0)
            test_end_time = test_start_time + td(hours=8)

            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes_BM(participant_df=data,
                                                                                           train_start_time=train_start_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"),
                                                                                           train_end_time=train_end_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"),
                                                                                           test_start_time=test_start_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"),
                                                                                           test_end_time=test_end_time.strftime(
                                                                                               "%m/%d/%Y %H:%M"))

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

            actuals_and_forecast_df = fit_multitarget_model_SPCI_BM(model=model, X_train=train_X, Y_train=train_y,
                                                            X_test=test_X, Y_test=test_y,
                                                            actuals_and_forecast_df=pd.DataFrame(),
                                                            targets=test_df.iloc[:,0:16].columns.values.tolist())

            results = pd.concat([results, actuals_and_forecast_df])
            start_time = start_time + td(hours=8)

        results.to_csv(path + ".csv", index=False)



    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()
        
        


def LEAR_alpha_BM(file_path: str):
    date_format = "%m/%d/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    dat = pd.read_csv(file_path, index_col="SettlementPeriod", parse_dates=True, date_parser=date_parse)

    dat = dat.drop(["index"], axis=1)
    dat = dat.bfill(axis='rows')
    dat = dat.ffill(axis='rows')
    dat = dat._get_numeric_data()

    Y = dat.iloc[:, 0:1]
    Y1 = dat.iloc[:, 0:16]
    X = dat.iloc[:, 16:]

    X_train_t = X.iloc[:7250, :]
    Y_train_t = Y1.iloc[:7250, :]

    rnn_train_1 = X_train_t.loc[:, "lag_-3x1": "lag_-50x1"]
    rnn_train_2 = X_train_t.loc[:, "lag_-3x2": "lag_-50x2"]
    rnn_train_3 = X_train_t.loc[:, "lag_-2x3":"lag_-49x3"]
    rnn_train_4 = X_train_t.loc[:, "lag_0x6": "lag_-47x6"]
    rnn_train_5 = X_train_t.loc[:, "lag_-2x12": "lag_-49x12"]

    rnn_train_6 = X_train_t.loc[:, "lag_2x7": "lag_17x7"]
    rnn_train_7 = X_train_t.loc[:, "lag_2x8": "lag_17x8"]
    rnn_train_8 = X_train_t.loc[:, "lag_2x9": "lag_17x9"]
    rnn_train_9 = X_train_t.loc[:, "lag_2x10":"lag_17x10"]
    rnn_train_10 = X_train_t.loc[:, "lag_2x11":"lag_17x11"]

    Y_scaler = preprocessing.MinMaxScaler()
    Y_train_Scaled = Y_scaler.fit_transform(Y_train_t)

    X_scalers = [preprocessing.MinMaxScaler() for _ in range(10)]
    rnn_scaled_train = [scaler.fit_transform(X_train) for scaler, X_train in zip(X_scalers, [rnn_train_1, rnn_train_2, rnn_train_3, rnn_train_4, rnn_train_5,
                                                                                                 rnn_train_6, rnn_train_7, rnn_train_8, rnn_train_9, rnn_train_10])]

    X_train_Scaled = np.concatenate(rnn_scaled_train, axis=1)

    alpha = LassoLarsIC(criterion='aic', max_iter=2500).fit(X_train_Scaled, Y_train_Scaled[:, :1].ravel()).alpha_
    return alpha

    
