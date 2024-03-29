# DAM and BM Forecasting and Trading

This repository contains code and datasets used for forecasting in both the Day-Ahead (DAM) market and the balancing market (BM).
Access to the full datasets can be found at: [Google Drive Link](https://drive.google.com/drive/u/0/folders/1GSJhwvhRZ5X5A0uJRZzkzCuJ8xu9kDcX)

### Balancing Market

For the Balancing Market (BM), we predict BM prices for the next 16 open settlement periods. The forecast horizon starts at \( t+2 \) as at time \( t \), the market periods \( t \) and \( t+1 \) are already closed, and adjustments can only be made from \( t+2 \), denoted as:

Y_{BM}=[BMP_{t+2},...,BMP_{t+17}].


In the BM dataset file this is presented as:

Y_{BM}=[lag_2y,...,lag_17y].


#### Historic data

The historical data used for BM price prediction includes:

- *BM Prices (BMP)*: BM prices from the most recent and available 24 hours: \( [BMP_{t-51},...,BMP_{t-3}] \).
- *BM Volume (BMV)*: The most recent 48 observations of BM volume: \( [BMV_{t-51},...,BMV_{t-3}] \).
- *Forecast Wind - Actual Wind (WDiff)*: Difference in forecast and actual wind data for the last 48 settlement periods: \( [WDiff_{t-50},...,WDiff_{t-2}] \).
- *Interconnector Values (I)*: Interconnector flows from the previous 24 hours: \( [I_{t-50},...,I_{t-2}] \).
- *DAM Prices (DAM)*: DAM prices from the previous 24 hours, used at an hourly granularity for each half-hour settlement period: \( [DAM_{t-48},...,DAM_t] \).

In the BM dataset file the headers are:

BMP=[lag_-3x1,...,lag_-51x1]; BMV=[lag_-3x2,...,lag_-51x2]; WDiff=[lag_-2x3,...,lag_-50x3]; I=[lag_-2x12,...,lag_-50x12]; DAM=[lag_0x6,...,lag_-47x6]

Unused Data:

Carbon Price Data - lag\_0x4

Gas Price Data - lag\_0x5

#### Forward/future-looking data

The future-looking data for BM price prediction includes:

- *Physical Notifications Volume (PHPN)*: The sum of physical notifications for the forecast horizon, spanning \( [PHPN_{t+2},...,PHPN_{t+17}] \).
- *Net Interconnector Schedule (PHI)*: Interconnector schedule for the forecast horizon: \( [PHI_{t+2},...,PHI_{t+17}] \).
- *Renewable Forecast (PHFW)*: TSO renewables forecast for non-dispatchable renewables for the forecast horizon: \( [PHFW_{t+2},...,PHFW_{t+17}] \).
- *Demand Forecast (PHFD)*: TSO demand forecast for the forecast horizon: \( [PHFD_{t+2},...,PHFD_{t+17}] \).
- *DAM Prices (DAM)*: DAM prices for the next 8 hours: \( [DAM_{t+1},...,DAM_{t+16}] \).

In the BM dataset file these headers are:

PHPN=[lag_2x7},...,lag_17x7]; PHI=[lag_2x8},...,lag_17x8]; PHFW=[lag_2x9},...,lag_17x9]; PHFD=[lag_2x10},...,lag_17x10]; DAM=[lag_2x11},...,lag_17x11]  

### Day-Ahead Market (DAM)
In our analysis, we focus on predicting DAM prices, with the forecast horizon extending to the subsequent 24 settlement periods.
The historical data considered for DAM price prediction includes DAM prices for the previous 168 hours and the wind and demand forecasts for the same interval. We then consider the TSO wind and demand forecasts for the forecasting horizon of 24 settlement periods.


The variation in time intervals for historical data is due to availability, limited to the most recent and accessible 48 observations from the data source. For further details on our forecasting approach, market structure, datasets, and variables for both the DAM and BM.


## Purpose
The purpose of this project is to analyze and compare different trading strategies of varying frequency in both the DAM and BM.

## Results
- Results for all DAM and BM quantile forecasts can be found in the DAM and BM folder. These can be used to test the trading strategies code.

## Code

### Quantile Forecasting Library

The 'Quantile Regression Library' directory contains the main code files used for forecasting. Below is a list of files included:
- Quantile Regression DAM BM.ipynb: This is the main notebook file containing code for running different forecasting models.
- LEAR_QR_DAM_BM.py: Python script containing functions related to LEAR Model.
- RF_LGBM_QR_DAM_BM.py: Python script containing functions related to Light Gradient Boosting Machines, and Random Forest Models Implementation.
- DNN_QR_DAM_BM.py: Python script containing functions related to Multi-Head Deep Neural Network modeling.

The 'Quantile Regression DAM BM.ipynb' notebook serves as the main file for generating Quantile Forecasts. It imports and utilizes the functions defined in the Python scripts mentioned above.

### Conformal Prediction Forecasting Library
The 'Conformal Prediction Forecasting Library' directory contains the main code files used for forecasting using EnbPI and SPCI. Below is a list of files included:
- Conformal Prediction For DAM & BM.ipynb: This is the main notebook file containing code for running different forecasting models.
- EnbPI_SPCI_BM.py: Python script containing functions related to BM Forecasts for All CP Models.
- EnbPI_SPCI_DAM.py: Python script containing functions related to DAM Forecasts for All CP Models.

The 'Conformal Prediction For DAM & BM.ipynb' notebook serves as the main file for generating Quantile Forecasts. It imports and utilizes the functions defined in the Python scripts mentioned above.

### Quantile Trading Strategies
This repository contains Python scripts and a Jupyter notebook implementing various quantile trading strategies for both the Day-Ahead Market (DAM) and Balancing Market (BM).

- Single_Trade.py: Python script with functions related to the Single Trade strategy (TS1) for both the DAM and BM.
- Multi_Trade.py: Python script with functions related to the Multi-Trade strategy (TS2) for both the DAM and BM.
- High_Frequency.py: Python script with functions related to the High-Frequency Trading strategy (TS3) for both the DAM and BM.
- Dual_Strategy.py: Python script containing functions related to the Dual Strategy (TS3-DUAL), which combines elements from both the DAM and BM trading strategies. This file includes functions for loading data, processing prices, implementing the dual strategy, calculating trading results, and printing results.
- Trading_Strategies.ipynb: This Jupyter notebook serves as the main file, containing code for implementing and analyzing the Single Trade, Multi-Trade, High-Frequency, and Dual-Strategy Trading strategies in both the DAM and BM markets.
  
These scripts and the notebook provide tools for implementing and experimenting with different trading strategies, allowing for analysis and comparison of their performance in both market contexts.
