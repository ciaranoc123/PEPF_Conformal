import os;
# os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from datetime import timedelta as td
import traceback
# from pandas.plotting import scatter_matrix
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
from keras.layers import Input
from keras.models import Model

# tensorflow &keras
import tensorflow as tf
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn import preprocessing as prep
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, make_scorer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, mean_absolute_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import tensorflow.keras.backend as K
import warnings
    
    
import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime as dt

def load_and_preprocess_data_DNN_DAM(file_path):
    # Define date format and parsing function
    date_format = "%d/%m/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    
    # Read the CSV file
    dat = pd.read_csv(file_path, index_col="DeliveryPeriod", parse_dates=True, date_parser=date_parse)
    
    # Handle missing values
    dat = dat.bfill(axis='rows')
    dat = dat.ffill(axis='rows')
    
    # Selecting relevant columns for RNN training
    rnn_train_1 = dat.loc[:, "EURPrices-24":"EURPrices-167"]
    rnn_train_2 = dat.loc[:, "WF": "WF-143"]
    rnn_train_3 = dat.loc[:, "DF": "DF-143"]
    
    # Scaling features
    X_scaler1 = preprocessing.MinMaxScaler()
    X_scaler2 = preprocessing.MinMaxScaler()
    X_scaler3 = preprocessing.MinMaxScaler()
    
    rnn_scaled_train_1 = X_scaler1.fit_transform(rnn_train_1)
    rnn_scaled_train_2 = X_scaler2.fit_transform(rnn_train_2)
    rnn_scaled_train_3 = X_scaler3.fit_transform(rnn_train_3)
    
    # Reshape data for RNN input
    X_train_Scaled = np.hstack((rnn_scaled_train_1, rnn_scaled_train_2, rnn_scaled_train_3)).reshape(rnn_train_1.shape[0], 3, 144).transpose(0, 2, 1)
    
    i_shape = (X_train_Scaled.shape[1], X_train_Scaled.shape[2])
    
    return dat


    
    

def generate_train_and_test_dataframes_DNN_DAM(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
                                       test_start_time: dt, test_end_time: dt):
    """
    This method takes the raw information contained in the participat_df (i.e. explanatory variables and targets) and produces dataframes
        train_X, train_y, test_X, test_y, test_df
    What are the uses of these dataframes?
        - The train_X and train_y dataframes can be used to train models.
        - For the trained model, predictions can then be made using the test_X dataframe.
        - Predictions made in the previous step can then be compared to actual/target values contained in the test_y dataframe.
        - Finally, the test_df is used by other methods for plotting.
    Thus, this method will be called repeatedly in the rolling_walk_forward_validation method/process.

    Parameters
    ----------
    participant_df : pd.DataFrame
        Pandas dataframe, contains the participant time series info (i.e. explanatory and target variables, the index will be te trading period).
    date_time_column : str
        This is the column in the participant_df which indicates the deliveryperiod.
    train_start_time : dt
        The train_X and train_y dataframes will cover the interval [train_start_time, train_end_time].
    train_end_time : dt
        See previous comment.
    test_start_time : dt
        The test_X and test_y dataframes will cover the 24 trading periods from [train_end_time, train_end_time + 24 hours].
    test_end_time : dt
        See previous comment.
    columns_to_exclude: [str]
        These are the columns participant_df which should be ignored i.e. columns we don't want to use as explanatory variables.
    features_to_encode: [str]
        These are the categorical columns for which we want to apply one hot encoding.
    prefix_to_include: [str]
        For the categorical columns to which we apply one hot encoding, this list helps inform the naming convention for the newly created columns.
    targets: [str]
        These are the columns that we are trying to predict.

    Returns
    -------
    A tuple of dataframes.
                train_X, train_y, test_X, test_y,test_df
    Details and use cases for these dataframes are described above.
    """

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
        rnn_test_Y = test_df.loc[:, "EURPrices":"EURPrices+23"]

        X_scaler1 = preprocessing.MinMaxScaler()
        X_scaler2 = preprocessing.MinMaxScaler()
        X_scaler3 = preprocessing.MinMaxScaler()
        Y_scaler = preprocessing.MinMaxScaler()

        rnn_scaled_train_1 = X_scaler1.fit_transform(rnn_train_1)
        rnn_scaled_train_2 = X_scaler2.fit_transform(rnn_train_2)
        rnn_scaled_train_3 = X_scaler3.fit_transform(rnn_train_3)

        train_y = Y_scaler.fit_transform(rnn_Y)
        Y_scaler_n = Y_scaler.fit(rnn_Y)

        Y_test_scaled = Y_scaler.transform(rnn_test_Y)

        train_X = np.hstack(
            (rnn_scaled_train_1, rnn_scaled_train_2, rnn_scaled_train_3)
        ).reshape(rnn_train_1.shape[0], 3, 144).transpose(0, 2, 1)

        test_X = np.hstack(
            (X_scaler1.transform(rnn_test_1), X_scaler2.transform(rnn_test_2), X_scaler3.transform(rnn_test_3))
        ).reshape(rnn_test_1.shape[0], 3, 144).transpose(0, 2, 1)

        test_y = test_df.iloc[:, 0:24]

        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n

    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n


def fit_multitarget_model_DNN_DAM(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets, Y_scaler_n):
    try:
        Y_scaler = preprocessing.MinMaxScaler()
        Y_scaler = Y_scaler.fit(Y_train)
        cols = Y_test.iloc[:, 0:24].columns.values.tolist() 
        es = EarlyStopping(monitor='val_loss', mode='min',  patience=30)

        model.fit(X_train, Y_train, epochs=3, verbose=0,  callbacks=[es], validation_split=0.1)
        model_test_predictions=None
        model_test_predictions = pd.DataFrame(Y_scaler_n.inverse_transform(np.array(model.predict(X_test).reshape(5,24))), columns=cols)


        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_10"] = model_test_predictions.iloc[:1, i].T.tolist() if len(
                cols) > 1 else model_test_predictions.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_30"] = model_test_predictions.iloc[1:2, i].T.tolist() if len(
                cols) > 1 else model_test_predictions.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_50"] = model_test_predictions.iloc[2:3, i].T.tolist() if len(
                cols) > 1 else model_test_predictions.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_70"] = model_test_predictions.iloc[3:4, i].T.tolist() if len(
                cols) > 1 else model_test_predictions.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[cols[i] + "_Forecast_90"] = model_test_predictions.iloc[4:, i].T.tolist() if len(
                cols) > 1 else model_test_predictions.tolist()

        return actuals_and_forecast_df

    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()


def rolling_walk_forward_validation_DNN_DAM(model, data, targets, start_time, end_time, training_days, path):
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
            train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n = generate_train_and_test_dataframes_DNN_DAM(
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
            actuals_and_forecast_df = fit_multitarget_model_DNN_DAM(model=model, Y_scaler_n=Y_scaler_n, X_train=train_X,
                                                            Y_train=train_y,
                                                            X_test=test_X, Y_test=test_y,
                                                            actuals_and_forecast_df=test_df.iloc[:, 0:24],
                                                            targets=test_df.iloc[:,0:24].columns.values.tolist())

            results = pd.concat([results, actuals_and_forecast_df])
            start_time = start_time + td(hours=24)

        results.to_csv(path + ".csv", index=False)



    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()


def create_model():
    def qloss(qs, y_true, y_pred):
        # Pinball loss for multiple quantiles
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return K.mean(v)

    loss_10th_p = lambda y_true, y_pred: qloss(0.1, y_true, y_pred)
    loss_30th_p = lambda y_true, y_pred: qloss(0.3, y_true, y_pred)
    loss_50th_p = lambda y_true, y_pred: qloss(0.5, y_true, y_pred)
    loss_70th_p = lambda y_true, y_pred: qloss(0.7, y_true, y_pred)
    loss_90th_p = lambda y_true, y_pred: qloss(0.9, y_true, y_pred)
    i_shape = (144, 3)

    net_input = Input(shape=i_shape)

    input_00 = Flatten(input_shape=i_shape)(net_input)
    input_10 = Flatten(input_shape=i_shape)(net_input)
    input_20 = Flatten(input_shape=i_shape)(net_input)
    input_30 = Flatten(input_shape=i_shape)(net_input)
    input_40 = Flatten(input_shape=i_shape)(net_input)

    for i in range(1):
        input_00 = Dense(24, activation='relu')(input_00)
    input_00 = Dropout(0.0)(input_00)

    for i in range(1):
        input_10 = Dense(24, activation='relu')(input_10)
    input_10 = Dropout(0.0)(input_10)

    for i in range(3):
        input_20 = Dense(24, activation='relu')(input_20)
    input_20 = Dropout(0.044444)(input_20)

    for i in range(3):
        input_30 = Dense(192, activation='sigmoid')(input_30)
    input_30 = Dropout(0.088889)(input_30)

    for i in range(1):
        input_40 = Dense(24, activation='sigmoid')(input_40)
    input_40 = Dropout(0.400000)(input_40)

    for i in range(2):
        input_00 = Dense(24, activation='sigmoid')(input_00)
    output_1 = Dense(24, name='out_10')(input_00)

    for i in range(1):
        input_10 = Dense(24, activation='sigmoid')(input_10)
    output_2 = Dense(24, name='out_20')(input_10)

    for i in range(1):
        input_20 = Dense(24, activation='sigmoid')(input_20)
    output_3 = Dense(24, name='out_30')(input_20)

    for i in range(1):
        input_30 = Dense(24, activation='sigmoid')(input_30)
    output_4 = Dense(24, name='out_40')(input_30)

    for i in range(2):
        input_40 = Dense(24, activation='relu')(input_40)
    output_5 = Dense(24, name='out_50')(input_40)

    opt = Adam(learning_rate=0.002311)
    model = Model(inputs=net_input, outputs=[output_1, output_2, output_3, output_4, output_5])
    model.compile(loss=[loss_10th_p, loss_30th_p, loss_50th_p, loss_70th_p, loss_90th_p], optimizer=opt)

    return model

mmo = KerasRegressor(build_fn=create_model, epochs=300, batch_size=32, verbose=0)






def generate_train_and_test_dataframes_DNN_BM(participant_df: pd.DataFrame, train_start_time: dt, train_end_time: dt, \
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

        
        rnn_train1_a = train_df.loc[:, "lag_-3x1":"lag_-18x1"]
        rnn_train1_b = train_df.loc[:, "lag_-19x1":"lag_-34x1"]
        rnn_train1_c = train_df.loc[:, "lag_-35x1":"lag_-50x1"]

        rnn_train2_a = train_df.loc[:, "lag_-3x2":"lag_-18x2"]
        rnn_train2_b = train_df.loc[:, "lag_-19x2":"lag_-34x2"]
        rnn_train2_c = train_df.loc[:, "lag_-35x2":"lag_-50x2"]

        rnn_train3_a = train_df.loc[:, "lag_-2x3":"lag_-17x3"]
        rnn_train3_b = train_df.loc[:, "lag_-18x3":"lag_-33x3"]
        rnn_train3_c = train_df.loc[:, "lag_-34x3":"lag_-49x3"]

        rnn_train4_a = train_df.loc[:, "lag_0x6":"lag_-15x6"]
        rnn_train4_b = train_df.loc[:, "lag_-16x6":"lag_-31x6"]
        rnn_train4_c = train_df.loc[:, "lag_-32x6":"lag_-47x6"]

        rnn_train5_a = train_df.loc[:, "lag_-2x12":"lag_-17x12"]
        rnn_train5_b = train_df.loc[:, "lag_-18x12":"lag_-33x12"]
        rnn_train5_c = train_df.loc[:, "lag_-34x12":"lag_-49x12"]

        rnn_train6 = train_df.loc[:, "lag_2x7":"lag_17x7"]
        rnn_train7 = train_df.loc[:, "lag_2x8":"lag_17x8"]
        rnn_train8 = train_df.loc[:, "lag_2x9":"lag_17x9"]
        rnn_train9 = train_df.loc[:, "lag_2x10":"lag_17x10"]
        rnn_train10 = train_df.loc[:, "lag_2x11":"lag_17x11"]

        rnn_test1_a = test_df.loc[:, "lag_-3x1":"lag_-18x1"]
        rnn_test1_b = test_df.loc[:, "lag_-19x1":"lag_-34x1"]
        rnn_test1_c = test_df.loc[:, "lag_-35x1":"lag_-50x1"]

        rnn_test2_a = test_df.loc[:, "lag_-3x2":"lag_-18x2"]
        rnn_test2_b = test_df.loc[:, "lag_-19x2":"lag_-34x2"]
        rnn_test2_c = test_df.loc[:, "lag_-35x2":"lag_-50x2"]

        rnn_test3_a = test_df.loc[:, "lag_-2x3":"lag_-17x3"]
        rnn_test3_b = test_df.loc[:, "lag_-18x3":"lag_-33x3"]
        rnn_test3_c = test_df.loc[:, "lag_-34x3":"lag_-49x3"]

        rnn_test4_a = test_df.loc[:, "lag_0x6":"lag_-15x6"]
        rnn_test4_b = test_df.loc[:, "lag_-16x6":"lag_-31x6"]
        rnn_test4_c = test_df.loc[:, "lag_-32x6":"lag_-47x6"]

        rnn_test5_a = test_df.loc[:, "lag_-2x12":"lag_-17x12"]
        rnn_test5_b = test_df.loc[:, "lag_-18x12":"lag_-33x12"]
        rnn_test5_c = test_df.loc[:, "lag_-34x12":"lag_-49x12"]

        rnn_test6 = test_df.loc[:, "lag_2x7":"lag_17x7"]
        rnn_test7 = test_df.loc[:, "lag_2x8":"lag_17x8"]
        rnn_test8 = test_df.loc[:, "lag_2x9":"lag_17x9"]
        rnn_test9 = test_df.loc[:, "lag_2x10":"lag_17x10"]
        rnn_test10 = test_df.loc[:, "lag_2x11":"lag_17x11"]

        rnn_Y = train_df.loc[:, "lag_2y":"lag_17y"]

        X_scaler1_a = preprocessing.MinMaxScaler()
        X_scaler1_b = preprocessing.MinMaxScaler()
        X_scaler1_c = preprocessing.MinMaxScaler()

        X_scaler2_a = preprocessing.MinMaxScaler()
        X_scaler2_b = preprocessing.MinMaxScaler()
        X_scaler2_c = preprocessing.MinMaxScaler()

        X_scaler3_a = preprocessing.MinMaxScaler()
        X_scaler3_b = preprocessing.MinMaxScaler()
        X_scaler3_c = preprocessing.MinMaxScaler()

        X_scaler4_a = preprocessing.MinMaxScaler()
        X_scaler4_b = preprocessing.MinMaxScaler()
        X_scaler4_c = preprocessing.MinMaxScaler()

        X_scaler5_a = preprocessing.MinMaxScaler()
        X_scaler5_b = preprocessing.MinMaxScaler()
        X_scaler5_c = preprocessing.MinMaxScaler()


        X_scaler6 = preprocessing.MinMaxScaler()
        X_scaler7 = preprocessing.MinMaxScaler()
        X_scaler8 = preprocessing.MinMaxScaler()
        X_scaler9 = preprocessing.MinMaxScaler()
        X_scaler10 = preprocessing.MinMaxScaler()

        Y_scaler = preprocessing.MinMaxScaler()

        rnn_scaled_train1_a = X_scaler1_a.fit_transform(rnn_train1_a)
        rnn_scaled_train1_b = X_scaler1_b.fit_transform(rnn_train1_b)
        rnn_scaled_train1_c = X_scaler1_c.fit_transform(rnn_train1_c)

        rnn_scaled_train2_a = X_scaler2_a.fit_transform(rnn_train2_a)
        rnn_scaled_train2_b = X_scaler2_b.fit_transform(rnn_train2_b)
        rnn_scaled_train2_c = X_scaler2_c.fit_transform(rnn_train2_c)

        rnn_scaled_train3_a = X_scaler3_a.fit_transform(rnn_train3_a)
        rnn_scaled_train3_b = X_scaler3_b.fit_transform(rnn_train3_b)
        rnn_scaled_train3_c = X_scaler3_c.fit_transform(rnn_train3_c)

        rnn_scaled_train4_a = X_scaler4_a.fit_transform(rnn_train4_a)
        rnn_scaled_train4_b = X_scaler4_b.fit_transform(rnn_train4_b)
        rnn_scaled_train4_c = X_scaler4_c.fit_transform(rnn_train4_c)

        rnn_scaled_train5_a = X_scaler5_a.fit_transform(rnn_train5_a)
        rnn_scaled_train5_b = X_scaler5_b.fit_transform(rnn_train5_b)
        rnn_scaled_train5_c = X_scaler5_c.fit_transform(rnn_train5_c)

        rnn_scaled_train6 = X_scaler6.fit_transform(rnn_train6)
        rnn_scaled_train7 = X_scaler7.fit_transform(rnn_train7)
        rnn_scaled_train8 = X_scaler8.fit_transform(rnn_train8)
        rnn_scaled_train9 = X_scaler9.fit_transform(rnn_train9)
        rnn_scaled_train10 = X_scaler10.fit_transform(rnn_train10)

        train_y   = Y_scaler.fit_transform(rnn_Y)
        Y_scaler_n=Y_scaler.fit(rnn_Y)
        train_X = np.hstack(
            (rnn_scaled_train1_a, rnn_scaled_train1_b, rnn_scaled_train1_c, rnn_scaled_train2_a, rnn_scaled_train2_b, rnn_scaled_train2_c,
             rnn_scaled_train3_a, rnn_scaled_train3_b, rnn_scaled_train3_c, rnn_scaled_train4_a, rnn_scaled_train4_b, rnn_scaled_train4_c,
             rnn_scaled_train5_a, rnn_scaled_train5_b, rnn_scaled_train5_c,rnn_scaled_train6,rnn_scaled_train7, rnn_scaled_train8,
             rnn_scaled_train9, rnn_scaled_train10)
        ).reshape(rnn_train6.shape[0], 20, 16).transpose(0, 2, 1)

        test_X = np.hstack(
            (X_scaler1_a.transform(rnn_test1_a),X_scaler1_b.transform(rnn_test1_b),X_scaler1_c.transform(rnn_test1_c),
             X_scaler2_a.transform(rnn_test2_a),X_scaler2_b.transform(rnn_test2_b),X_scaler2_c.transform(rnn_test2_c),
             X_scaler3_a.transform(rnn_test3_a),X_scaler3_b.transform(rnn_test3_b),X_scaler3_c.transform(rnn_test3_c),
             X_scaler4_a.transform(rnn_test4_a),X_scaler4_b.transform(rnn_test4_b),X_scaler4_c.transform(rnn_test4_c),
             X_scaler5_a.transform(rnn_test5_a),X_scaler5_b.transform(rnn_test5_b),X_scaler5_c.transform(rnn_test5_c),
             X_scaler6.transform(rnn_test6),X_scaler7.transform(rnn_test7),X_scaler8.transform(rnn_test8),
             X_scaler9.transform(rnn_test9), X_scaler10.transform(rnn_test10))
        ).reshape(rnn_test6.shape[0], 20, 16).transpose(0, 2, 1)
        
        test_y = test_df.iloc[:, 0:16]
                                
        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n
        
    except Exception:
        print("Error: generate_train_and_test_dataframes method.")
        traceback.print_exc()
        return train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n


def fit_multitarget_model_DNN_BM(model, X_train, Y_train, X_test, Y_test, actuals_and_forecast_df, targets, Y_scaler_n):
  
    try:
        Y_scaler = preprocessing.MinMaxScaler()
        Y_scaler = Y_scaler.fit(Y_train)
        cols = pd.DataFrame(Y_train).columns.values.tolist()                     
        es = EarlyStopping(monitor='val_loss', mode='min',  patience=30)

        model.fit(X_train, Y_train,epochs=3, verbose=0, callbacks=[es], validation_split=0.1) 
        model_test_predictions=None  
        model_test_predictions = pd.DataFrame(Y_scaler_n.inverse_transform(model.predict(X_test).reshape(5,16)), columns=cols)

        
        for i in range(0, len(cols)):
            actuals_and_forecast_df[str(cols[i]) + "_Forecast_10"] = model_test_predictions.iloc[:1, i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[str(cols[i]) + "_Forecast_30"] = model_test_predictions.iloc[1:2, i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[str(cols[i]) + "_Forecast_50"] = model_test_predictions.iloc[2:3, i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[str(cols[i]) + "_Forecast_70"] = model_test_predictions.iloc[3:4, i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist()

        for i in range(0, len(cols)):
            actuals_and_forecast_df[str(cols[i]) + "_Forecast_90"] = model_test_predictions.iloc[4:5, i].T.tolist() if len(cols) > 1 else model_test_predictions.tolist()
            
            
        return actuals_and_forecast_df
    
    except Exception:
        print("Error: fit_multitarget_model method.")
        traceback.print_exc()
        return pd.DataFrame()
  
    
def rolling_walk_forward_validation_DNN_BM(model, data, targets, start_time, end_time, training_days, path):
 
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
            train_X, train_y, test_X, test_y, test_df, train_df, Y_scaler_n= generate_train_and_test_dataframes_DNN_BM(participant_df=data, train_start_time=train_start_time.strftime("%m/%d/%Y %H:%M"), train_end_time=train_end_time.strftime("%m/%d/%Y %H:%M"), 
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
            actuals_and_forecast_df = fit_multitarget_model_DNN_BM(model=model,Y_scaler_n=Y_scaler_n, X_train=train_X, Y_train=train_y,
                                            X_test=test_X, Y_test=test_y, actuals_and_forecast_df=test_df.iloc[:,0:16], targets=test_df.iloc[:,0:16].columns.values.tolist())

    
            results = pd.concat([results, actuals_and_forecast_df])
        
            start_time = start_time + td(hours=8)
            
        results.to_csv(path  + ".csv", index = False)
        

        
    except Exception:
        print("Error: rolling_walk_forward_validation method.")
        traceback.print_exc()




def create_model_BM():    
    def qloss(qs, y_true, y_pred):
        # Pinball loss for multiple quantiles
        q = tf.constant(np.array([qs]), dtype=tf.float32)
        e = y_true - y_pred
        v = tf.maximum(q * e, (q - 1) * e)
        return K.mean(v)

    loss_10th_p = lambda y_true, y_pred: qloss(0.1, y_true, y_pred)
    loss_30th_p = lambda y_true, y_pred: qloss(0.3, y_true, y_pred)
    loss_50th_p = lambda y_true, y_pred: qloss(0.5, y_true, y_pred)
    loss_70th_p = lambda y_true, y_pred: qloss(0.7, y_true, y_pred)
    loss_90th_p = lambda y_true, y_pred: qloss(0.9, y_true, y_pred)
    
    i_shape = (16, 20)    
    net_input = Input(shape=i_shape)    
    
    input_00 = Flatten(input_shape=i_shape)(net_input)
    input_10 = Flatten(input_shape=i_shape)(net_input)
    input_20 = Flatten(input_shape=i_shape)(net_input)
    input_30 = Flatten(input_shape=i_shape)(net_input)
    input_40 = Flatten(input_shape=i_shape)(net_input)
    
    for i in range(3):
        input_00 = Dense(64, activation='relu')(input_00) 
    input_00 = Dropout(0.0)(input_00)
       
    for i in range(2):
        input_10 = Dense(192, activation='relu')(input_10) 
    input_10 = Dropout(0.222222)(input_10)

    for i in range(1):
        input_20 = Dense(192, activation='relu')(input_20)  
    input_20 = Dropout(0.044444)(input_20)
        
    for i in range(3):
        input_30 = Dense(32, activation='relu')(input_30) 
    input_30 = Dropout(0.355556)(input_30)
        
    for i in range(3):
        input_40 = Dense(256, activation='relu')(input_40) 
    input_40 = Dropout(0.0)(input_40)
        

       
    for i in range(2):
        input_00 = Dense(256, activation='relu')(input_00) 
    output_1 = Dense(16, name='out_10')(input_00)
    
    for i in range(2):
        input_10 = Dense(128, activation='relu')(input_10) 
    output_2 = Dense(16, name='out_20')(input_10)

    for i in range(2):
        input_20 = Dense(128, activation='relu')(input_20) 
    output_3 = Dense(16, name='out_30')(input_20)
    
    for i in range(1):
        input_30 = Dense(16, activation='relu')(input_30) 
    output_4 = Dense(16, name='out_40')(input_30)
    
    for i in range(2):
        input_40 = Dense(128, activation='relu')(input_40) 
    output_5 = Dense(16, name='out_50')(input_40)
    
    opt = Adam(learning_rate=0.006733)
    model = Model(inputs=net_input, outputs=[output_1, output_2, output_3, output_4, output_5])    
    model.compile(loss=[loss_10th_p, loss_30th_p, loss_50th_p, loss_70th_p, loss_90th_p], optimizer=opt)
    
    return model

mmo_BM = KerasRegressor(build_fn=create_model_BM, epochs=3, batch_size=8, verbose=2)


