import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt



def load_data_DAM(file_path):
    # Load data
    dat = pd.read_csv(file_path)
    dat1 = pd.DataFrame(dat)
    dat1 = dat1.iloc[152:, :].reset_index(drop=True)

    # Create quantile dataframes for different forecast quantiles (10%, 30%, 50%, 70%, 90%)
    column_names = ['EURPrices+{}_Forecast_10'.format(i) for i in range(0, 24)]
    Q_10 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['EURPrices+{}_Forecast_30'.format(i) for i in range(0, 24)]
    Q_30 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['EURPrices+{}_Forecast_50'.format(i) for i in range(0, 24)]
    Q_50 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['EURPrices+{}_Forecast_70'.format(i) for i in range(0, 24)]
    Q_70 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['EURPrices+{}_Forecast_90'.format(i) for i in range(0, 24)]
    Q_90 = dat1[column_names].dropna().stack().reset_index()

    # Create a dataframe 'Y_r' with price data extracted from specific columns
    column_names = ['EURPrices+{}'.format(i) for i in range(0, 24)]
    Y_r = dat1[column_names].dropna().stack().reset_index()
    Y_r = Y_r.iloc[:, :]

    # Set the "Price" column in each dataframe for later calculations
    Y_r["Price"] = Y_r.iloc[:, 2:3]
    Q_10["Price"] = Q_10.iloc[:, 2:3]
    Q_30["Price"] = Q_30.iloc[:, 2:3]
    Q_50["Price"] = Q_50.iloc[:, 2:3]
    Q_70["Price"] = Q_70.iloc[:, 2:3]
    Q_90["Price"] = Q_90.iloc[:, 2:3]

    return Y_r, Q_10, Q_30, Q_50, Q_70, Q_90
    
    
    


def run_electricity_strategy_DAM(df, Q_A_Preds, Q_B_Preds, eff_1, eff_2):
    # Create an empty list to store trading results
    prices = []
    # Get unique day indexes from the data
    day_index = df['level_0'].unique()

    # Loop through each day in the dataset.
    for day in day_index:
        # Filter the current day's data
        current_df = df[df['level_0'] == day]
        # Select the predicted prices for quantile A (Q_A) (alpha) for the current day
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == day]
        # Select the predicted prices for complementary quantile B (Q_B) (1-alpha) for the current day
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == day]
        
        # Find the maximum predicted price for the day
        max_price_index = current_Q_A['Price'].idxmax()
        # Establish all the remaining prices for that day that fall before the max price (these will be used to find the min price for pair 1)
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index)]
        
        # Find the minimum predicted price for the day
        min_price_index = current_Q_B['Price'].idxmin()
        # Establish all the remaining prices for that day that fall after the min price (these will be used to find the max price for pair 2)
        prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > min_price_index)]

        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        min_price_index1 = None
        max_price_index1 = None
        
        # Identifying the min price for the remaining prices that fall before the max
        if len(prices_before_max) > 0:
            min_price_index1 = prices_before_max['Price'].idxmin()
            
        # Identifying the max price for the remaining prices that fall after the min
        if len(prices_after_min) > 0:
            max_price_index1 = prices_after_min['Price'].idxmax()
            
        # Dealing with potential missing values (i.e., if the first min price is index 24, there will be no min price and vice versa)
        if max_price_index is not None and min_price_index1 is not None and max_price_index1 is not None:
            # All three indices are available; choosing the pair with the greater difference
            if (current_Q_A.loc[max_price_index, 'Price'] - current_Q_B.loc[min_price_index1, 'Price']) > (current_Q_A.loc[max_price_index1, 'Price'] - current_Q_B.loc[min_price_index, 'Price']):
                # Max price is chosen as the maximum in the remaining data, and min price is chosen as the minimum in the remaining data
                chosen_max_price_index = max_price_index
                chosen_min_price_index = min_price_index1
            else:
                # Min price is chosen as the minimum in the remaining data, and max price is chosen as the maximum in the remaining data
                chosen_max_price_index = max_price_index1
                chosen_min_price_index = min_price_index
        elif max_price_index is not None and min_price_index1 is not None:
            # max_price_index and min_price_index1 are available; choosing this pair
            chosen_max_price_index = max_price_index
            chosen_min_price_index = min_price_index1
        elif max_price_index1 is not None and min_price_index is not None:
            # max_price_index1 and min_price_index are available; choosing this pair
            chosen_max_price_index = max_price_index1
            chosen_min_price_index = min_price_index
            
        # Simple profit calculation using the identified timestamps from the predictions
        # Factoring in charge and discharge efficiency
        if chosen_max_price_index in current_df.index and chosen_min_price_index in current_df.index:
            profit = ((current_df.loc[chosen_max_price_index, 'Price']) * eff_1) - ((current_df.loc[chosen_min_price_index, 'Price']) / eff_2)
            prices.append((chosen_min_price_index, current_df.loc[chosen_min_price_index, 'Price'], chosen_max_price_index, current_df.loc[chosen_max_price_index, 'Price'], profit))
    
    # Create a DataFrame with trading results
    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice', 'maxPriceIndex', 'maxPrice', 'profit'])
    
    
def calculate_trading_results_DAM(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_dam_50_50 = run_electricity_strategy_DAM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2)
    r_dam_10_30 = run_electricity_strategy_DAM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=eff_1, eff_2=eff_2)
    r_dam_30_50 = run_electricity_strategy_DAM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2)
    r_dam_50_70 = run_electricity_strategy_DAM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2)
    r_dam_70_90 = run_electricity_strategy_DAM(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2)
    r_dam_30_70 = run_electricity_strategy_DAM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2)
    r_dam_10_90 = run_electricity_strategy_DAM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2)
    
    PF_DAM = run_electricity_strategy_DAM(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=1, eff_2=1)
    
    results = {
        'r_dam_50_50': np.round(sum(r_dam_50_50.iloc[:, 4:5].values), 2),
        'r_dam_10_30': np.round(sum(r_dam_10_30.iloc[:, 4:5].values), 2),
        'r_dam_30_50': np.round(sum(r_dam_30_50.iloc[:, 4:5].values), 2),
        'r_dam_50_70': np.round(sum(r_dam_50_70.iloc[:, 4:5].values), 2),
        'r_dam_70_90': np.round(sum(r_dam_70_90.iloc[:, 4:5].values), 2),
        'r_dam_30_70': np.round(sum(r_dam_30_70.iloc[:, 4:5].values), 2),
        'r_dam_10_90': np.round(sum(r_dam_10_90.iloc[:, 4:5].values), 2),
        'PF_DAM': np.round(sum(PF_DAM.iloc[:, 4:5].values), 2)
    }
    
    return results

def print_results_DAM(results):
    print("Trading results for different quantile pairs in the DAM:")
    for key, value in results.items():
        if key.startswith('r_dam'):
            quantiles = key.split('_')[2:]
            label = f"{quantiles[0]}-{quantiles[1]}"
            print(f"Total sum for trading quantile {label} pair in the DAM is: {value}")
        elif key == 'PF_DAM':
            print(f"Total sum for the Perfect Forecast pair in the DAM is: {value}")
            
            
            
            
            
         
         
         
         
            
            

def load_bm_data(file_path):
    # Define date format and parsing function
    date_format = "%m/%d/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    
    # Read the CSV file
    dat1 = pd.read_csv(file_path)
    dat1 = pd.DataFrame(dat1)
    dat1 = dat1.iloc[456:, :].reset_index(drop=True)

    # Create quantile dataframes for different forecast quantiles (10%, 30%, 50%, 70%, 90%)
    column_names = ['lag_{}y_Forecast_10'.format(i) for i in range(2, 18)]
    Q_10 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y_Forecast_30'.format(i) for i in range(2, 18)]
    Q_30 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y_Forecast_50'.format(i) for i in range(2, 18)]
    Q_50 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y_Forecast_70'.format(i) for i in range(2, 18)]
    Q_70 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y_Forecast_90'.format(i) for i in range(2, 18)]
    Q_90 = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y'.format(i) for i in range(2, 18)]
    Y_r = dat1[column_names].dropna().stack().reset_index()

    # Set the "Price" column in each dataframe for later calculations
    Y_r["Price"] = Y_r.iloc[:, 2:3]
    Q_10["Price"] = Q_10.iloc[:, 2:3]
    Q_30["Price"] = Q_30.iloc[:, 2:3]
    Q_50["Price"] = Q_50.iloc[:, 2:3]
    Q_70["Price"] = Q_70.iloc[:, 2:3]
    Q_90["Price"] = Q_90.iloc[:, 2:3]

    return Y_r, Q_10, Q_30, Q_50, Q_70, Q_90
            
    
    
def run_electricity_strategy_BM(df, Q_A_Preds, Q_B_Preds, eff_1, eff_2):
    # Create an empty list to store trading results
    prices = []
    # Get unique 8 hour period indexes from the data
    period_index = df['level_0'].unique()

    # Loop through each 8 hour period in the dataset.
    for period in period_index:
        # Filter the current 8 hour periods data
        current_df = df[df['level_0'] == period]
        # Select the predicted prices for quantile A (Q_A) (alpha) for the current 8 hour period
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == period]
        # Select the predicted prices for complementary quantile B (Q_B) (1-alpha) for the current 8 hour period
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == period]
        
        # Find the maximum predicted price for the 8 hour period
        max_price_index = current_Q_A['Price'].idxmax()
        # Establish all the remaining prices for that 8 hour period that fall before the max price (these will be used to find the min price for pair 1)
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == period) & (Q_B_Preds.index < max_price_index)]
        
        # Find the minimum predicted price for the 8 hour period
        min_price_index = current_Q_B['Price'].idxmin()
        # Establish all the remaining prices for that 8 hour period that fall after the min price (these will be used to find the max price for pair 2)
        prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == period) & (Q_A_Preds.index > min_price_index)]

        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        min_price_index1 = None
        max_price_index1 = None
        
        # Identifying the min price for the remaining prices that fall before the max
        if len(prices_before_max) > 0:
            min_price_index1 = prices_before_max['Price'].idxmin()
            
        # Identifying the max price for the remaining prices that fall after the min
        if len(prices_after_min) > 0:
            max_price_index1 = prices_after_min['Price'].idxmax()
            
        # Dealing with potential missing values (i.e., if the first min price is index 16, there will be no min price and vice versa)
        if max_price_index is not None and min_price_index1 is not None and max_price_index1 is not None:
            # All three indices are available; choosing the pair with the greater difference
            if (current_Q_A.loc[max_price_index, 'Price'] - current_Q_B.loc[min_price_index1, 'Price']) > (current_Q_A.loc[max_price_index1, 'Price'] - current_Q_B.loc[min_price_index, 'Price']):
                # Max price is chosen as the maximum in the remaining data, and min price is chosen as the minimum in the remaining data
                chosen_max_price_index = max_price_index
                chosen_min_price_index = min_price_index1
            else:
                # Min price is chosen as the minimum in the remaining data, and max price is chosen as the maximum in the remaining data
                chosen_max_price_index = max_price_index1
                chosen_min_price_index = min_price_index
        elif max_price_index is not None and min_price_index1 is not None:
            # max_price_index and min_price_index1 are available; choosing this pair
            chosen_max_price_index = max_price_index
            chosen_min_price_index = min_price_index1
        elif max_price_index1 is not None and min_price_index is not None:
            # max_price_index1 and min_price_index are available; choosing this pair
            chosen_max_price_index = max_price_index1
            chosen_min_price_index = min_price_index
            
        # Simple profit calculation using the identified timestamps from the predictions
        # Factoring in charge and discharge efficiency
        if chosen_max_price_index in current_df.index and chosen_min_price_index in current_df.index:
            profit = ((current_df.loc[chosen_max_price_index, 'Price']) * eff_1) - ((current_df.loc[chosen_min_price_index, 'Price']) / eff_2)
            prices.append((chosen_min_price_index, current_df.loc[chosen_min_price_index, 'Price'], chosen_max_price_index, current_df.loc[chosen_max_price_index, 'Price'], profit))
    
    # Create a DataFrame with trading results
    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice', 'maxPriceIndex', 'maxPrice', 'profit'])    
    
    



def calculate_bm_trading_results(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_bm_50_50 = run_electricity_strategy_BM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2)
    r_bm_10_30 = run_electricity_strategy_BM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=eff_1, eff_2=eff_2)
    r_bm_30_50 = run_electricity_strategy_BM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2)
    r_bm_50_70 = run_electricity_strategy_BM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2)
    r_bm_70_90 = run_electricity_strategy_BM(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2)
    r_bm_30_70 = run_electricity_strategy_BM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2)
    r_bm_10_90 = run_electricity_strategy_BM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2)
    
    PF_BM = run_electricity_strategy_BM(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=1, eff_2=1)
    
    results = {
        'r_bm_50_50': np.round(sum(r_bm_50_50.iloc[:, 4:5].values), 2),
        'r_bm_10_30': np.round(sum(r_bm_10_30.iloc[:, 4:5].values), 2),
        'r_bm_30_50': np.round(sum(r_bm_30_50.iloc[:, 4:5].values), 2),
        'r_bm_50_70': np.round(sum(r_bm_50_70.iloc[:, 4:5].values), 2),
        'r_bm_70_90': np.round(sum(r_bm_70_90.iloc[:, 4:5].values), 2),
        'r_bm_30_70': np.round(sum(r_bm_30_70.iloc[:, 4:5].values), 2),
        'r_bm_10_90': np.round(sum(r_bm_10_90.iloc[:, 4:5].values), 2),
        'PF_BM': np.round(sum(PF_BM.iloc[:, 4:5].values), 2)
    }
    
    return results

def print_results_BM(results):
    print("Trading results for different quantile pairs in the BM:")
    for key, value in results.items():
        if key.startswith('r_bm'):
            quantiles = key.split('_')[2:]
            label = f"{quantiles[0]}-{quantiles[1]}"
            print(f"Total sum for trading quantile {label} pair in the BM is: {value}")
        elif key == 'PF_BM':
            print(f"Total sum for the Perfect Forecast pair in the BM is: {value}")

            








# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_single_trade_BM(Y_r_bm, Q_A_Preds, Q_B_Preds, eff_1, eff_2):
    # Run electricity strategy for BM
    single_trade_bm = run_electricity_strategy_BM(df=Y_r_bm, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(single_trade_bm.iloc[:, 4:5].values, marker='o', linestyle='-', color='r', label='Profit')
    plt.title('Profit Obtained from Single-Trade Strategy (TS1-BM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()

# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_single_trade_DAM(Y_r, Q_A_Preds, Q_B_Preds, eff_1, eff_2):
    # Run electricity strategy for BM
    single_trade_dam = run_electricity_strategy_DAM(df=Y_r, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(single_trade_dam.iloc[:, 4:5].values, marker='o', linestyle='-', color='b', label='Profit')
    plt.title('Profit Obtained from Single-Trade Strategy (TS1-DAM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()



    

