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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import SPCI_class as SPCI
import torch

import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

date_format = "%d/%m/%Y %H:%M"
date_parse = lambda date: dt.datetime.strptime(date, date_format)
dat = pd.read_csv("/home/coconnor/DAM_VAR_1-3.csv", index_col="DeliveryPeriod", parse_dates=True,
                  date_parser=date_parse)

dat = pd.DataFrame(dat)
dat = dat.bfill(axis='rows')
dat = dat.ffill(axis='rows')

Y = dat.iloc[:, 0:1]
Y1 = dat.iloc[:, 0:24]
Y2 = dat.iloc[:, 0:24]
X = dat.iloc[:, 24:]
X_train = X.iloc[:12360, :]
Y_train = Y2.iloc[:12360, :]
X_test = X.iloc[12360:13104, :]
Y_test = Y2.iloc[12360:13104, :]

rnn_train_1 = X_train.loc[:, "EURPrices-24":"EURPrices-167"]
rnn_train_2 = X_train.loc[:, "WF": "WF-143"]
rnn_train_3 = X_train.loc[:, "DF": "DF-143"]

rnn_test_1 = X_test.loc[:, "EURPrices-24":"EURPrices-167"]
rnn_test_2 = X_test.loc[:, "WF": "WF-143"]
rnn_test_3 = X_test.loc[:, "DF": "DF-143"]

rnn_Y = Y_train.loc[:, "EURPrices":"EURPrices+23"]
rnn_test_Y = Y_test.loc[:, "EURPrices":"EURPrices+23"]

X_scaler1 = preprocessing.MinMaxScaler()
X_scaler2 = preprocessing.MinMaxScaler()
X_scaler3 = preprocessing.MinMaxScaler()
Y_scaler = preprocessing.MinMaxScaler()

rnn_scaled_train_1 = X_scaler1.fit_transform(rnn_train_1)
rnn_scaled_train_2 = X_scaler2.fit_transform(rnn_train_2)
rnn_scaled_train_3 = X_scaler3.fit_transform(rnn_train_3)

Y_train_Scaled = Y_scaler.fit_transform(rnn_Y)
Y_test_scaled = Y_scaler.transform(rnn_test_Y)

X_train_Scaled = np.concatenate((rnn_scaled_train_1, rnn_scaled_train_2, rnn_scaled_train_3), axis=1)

alpha = LassoLarsIC(criterion='aic', max_iter=2500).fit(X_train_Scaled, Y_train_Scaled[:, :1].ravel()).alpha_


def generate_train_and_test_dataframes(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
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

        train_X = train_df.iloc[:, 24:]
        test_X = test_df.iloc[:, 24:]
        train_y = train_df.iloc[:, 0:1]
        test_y = test_df.iloc[:, 0:1]

        return train_X, train_y, test_X, test_y, test_df

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df


def fit_multitarget_model(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets):
    try:
        cols = Y1.columns.values.tolist()
        model.fit(X_train, Y_train)
        model_test_predictions = None
        model_test_predictions = model.predict(X_test)
        model_test_predictions_5 = np.array(model_test_predictions)
        model_test_predictions_5 = model_test_predictions_5.reshape(24, 1)

        X_train_t1 = torch.from_numpy(np.array(X_train))
        X_test_t1 = torch.from_numpy(np.array(X_test))
        Y_train_t1 = torch.from_numpy(np.array(Y_train))
        Y_test_t1 = torch.from_numpy(np.array(Y_test))

        X_train_t2 = torch.from_numpy(np.array(X_train))
        X_test_t2 = torch.from_numpy(np.array(X_test))
        Y_train_t2 = torch.from_numpy(np.array(Y_train))
        Y_test_t2 = torch.from_numpy(np.array(Y_test))

        fit_func = model
        stride = 24
        SPCI_class_1 = SPCI.SPCI_and_EnbPI(X_train_t1, X_test_t1, Y_train_t1, Y_test_t1, fit_func=fit_func)
        SPCI_class_1.fit_bootstrap_models_online_multistep(B=10, fit_sigmaX=False, stride=stride)
        use_SPCI = True
        smallT = not use_SPCI
        SPCI_class_1.compute_PIs_Ensemble_online(0.1, smallT=smallT, past_window=365, use_SPCI=use_SPCI,
                                                 quantile_regr=True, stride=stride)
        Pred_ints_1 = SPCI_class_1.PIs_Ensemble
        model_test_predictions_1 = np.array(Pred_ints_1)
        model_test_predictions_1 = model_test_predictions_1.reshape(24, 2)

        SPCI_class_2 = SPCI.SPCI_and_EnbPI(X_train_t2, X_test_t2, Y_train_t2, Y_test_t2, fit_func=fit_func)
        SPCI_class_2.fit_bootstrap_models_online_multistep(B=10, fit_sigmaX=False, stride=stride)
        use_SPCI = True
        smallT = not use_SPCI
        SPCI_class_2.compute_PIs_Ensemble_online(0.3, smallT=smallT, past_window=365, use_SPCI=use_SPCI,
                                                 quantile_regr=True, stride=stride)
        Pred_ints_2 = SPCI_class_2.PIs_Ensemble
        model_test_predictions_2 = np.array(Pred_ints_2)
        model_test_predictions_2 = model_test_predictions_2.reshape(24, 2)

        print(model_test_predictions_1)

        for i in range(0, len(cols)):
            actuals_and_forecast_df["EURPrices" + cols[i]] = np.array(Y_test_t1[i, :]).tolist() if len(
                cols) > 1 else np.array(Y_test_t1).tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_10"] = model_test_predictions_1[i, 0:1].tolist() if len(
                cols) > 1 else model_test_predictions_1.tolist()
        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_10"] = model_test_predictions_1[i, 0:1].tolist() if len(
                cols) > 1 else model_test_predictions_1.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_30"] = model_test_predictions_2[i, 0:1].tolist() if len(
                cols) > 1 else model_test_predictions_2.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_50"] = model_test_predictions_5[i, 0:1].tolist() if len(
                cols) > 1 else model_test_predictions_5.tolist()

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


def rolling_walk_forward_validation(model, data, targets, start_time, end_time, training_days, path):
    try:

        all_columns = list(data.columns)
        results = pd.DataFrame()

        date_format = "%d/%m/%Y %H:%M"
        start_time = dt.datetime.strptime(start_time, date_format)
        end_time = dt.datetime.strptime(end_time, date_format)

        while start_time < end_time:

            train_start_time = start_time + td(days=training_days)
            train_end_time = start_time

            test_start_time = train_end_time + td(hours=0)
            test_end_time = test_start_time + td(hours=24)

            print("train_start_time: " + str(train_start_time) + ", train_end_time: " + str(train_end_time) + \
                  ", test_start_time: " + str(test_start_time) + ", test_end_time: " + str(test_end_time))

            train_X, train_y, test_X, test_y, test_df = generate_train_and_test_dataframes(participant_df=dat,
                                                                                           train_start_time=train_start_time.strftime(
                                                                                               "%d/%m/%Y %H:%M"),
                                                                                           train_end_time=train_end_time.strftime(
                                                                                               "%d/%m/%Y %H:%M"),
                                                                                           test_start_time=test_start_time.strftime(
                                                                                               "%d/%m/%Y %H:%M"),
                                                                                           test_end_time=test_end_time.strftime(
                                                                                               "%d/%m/%Y %H:%M"))

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

            actuals_and_forecast_df = fit_multitarget_model(model=model, X_train=train_X, Y_train=train_y,
                                                            X_test=test_X,
                                                            Y_test=test_y,
                                                            actuals_and_forecast_df=pd.DataFrame(),
                                                            targets=Y1.columns.values.tolist())

            results = results.append(actuals_and_forecast_df)
            print(results)
            start_time = start_time + td(hours=24)

        results.to_csv(path + ".csv", index=False)



    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()


rolling_walk_forward_validation(model=Lasso(max_iter=2500, alpha=alpha),
                                data=dat, start_time='1/6/2020 00:00', end_time='1/9/2020  00:00',
                                targets=Y.columns.values.tolist(), training_days=-210,
                                path="/home/coconnor/LEAR_SPCI_DAM_1-3")