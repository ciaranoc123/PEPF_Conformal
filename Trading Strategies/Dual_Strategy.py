import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt



# # Create a DataFrame with a range of rows and a 'level_0' column that repeats values from 0 to 365, 48 times each (adjusting for the specified number of rows).
# # This is to ensure the DAM and BM 'level_0' used to identify each period now match. 
# # BM will no longer have 3 separate periods, but one single one, as DAM trades will dictate split.

def load_bm_data_DS(file_path):
    num_rows = 365 * 48
    df = pd.DataFrame(index=range(num_rows))
    df['level_0'] = np.repeat(np.arange(366), 48)[:num_rows]
# # Read data from BM CSV file.

    date_format = "%m/%d/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
# # Concatenate data from the CSV file with the previously created DataFrame to have 365 different 'level_0' values.
    # # DAM and BM 'level_0' values will now match

    dat1 = pd.read_csv(file_path)
    dat1 = pd.DataFrame(dat1)
    dat1 = dat1.iloc[456:, :].reset_index(drop=True)
    dat1 = pd.concat([dat1, df], axis=1)
# # Create quantile dataframes for different forecast quantiles (10%, 30%, 50%, 70%, 90%)
# # These dataframes are extracted from columns with specific names in the 'dat1' dataframe
    column_names = ['lag_{}y_Forecast_10'.format(i) for i in range(2, 18)]
    Q_10_BM = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y_Forecast_30'.format(i) for i in range(2, 18)]
    Q_30_BM = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y_Forecast_50'.format(i) for i in range(2, 18)]
    Q_50_BM = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y_Forecast_70'.format(i) for i in range(2, 18)]
    Q_70_BM = dat1[column_names].dropna().stack().reset_index()
    column_names = ['lag_{}y_Forecast_90'.format(i) for i in range(2, 18)]
    Q_90_BM = dat1[column_names].dropna().stack().reset_index()
# # Create a dataframe 'Y_r' with price data extracted from specific columns
# # These dataframes are extracted from columns with specific names in the 'dat1' dataframe
    column_names = ['lag_{}y'.format(i) for i in range(2, 18)]
    Y_r_BM = dat1[column_names].dropna().stack().reset_index()
    Y_r_BM["Price"] = Y_r_BM.iloc[:, 2:3]
    Q_10_BM["Price"] = Q_10_BM.iloc[:, 2:3]
    Q_30_BM["Price"] = Q_30_BM.iloc[:, 2:3]
    Q_50_BM["Price"] = Q_50_BM.iloc[:, 2:3]
    Q_70_BM["Price"] = Q_70_BM.iloc[:, 2:3]
    Q_90_BM["Price"] = Q_90_BM.iloc[:, 2:3]
# # Set 'Price' columns for each DataFrame in BM.

    Y_r_BM = Y_r_BM.iloc[:, 1:]
    Q_10_BM = Q_10_BM.iloc[:, 1:]
    Q_30_BM = Q_30_BM.iloc[:, 1:]
    Q_50_BM = Q_50_BM.iloc[:, 1:]
    Q_70_BM = Q_70_BM.iloc[:, 1:]
    Q_90_BM = Q_90_BM.iloc[:, 1:]
# # Concatenate each DataFrame with the previously created 'df' DataFrame.

    Y_r_BM = pd.concat([Y_r_BM, df], axis=1)
    Q_10_BM = pd.concat([Q_10_BM, df], axis=1)
    Q_30_BM = pd.concat([Q_30_BM, df], axis=1)
    Q_50_BM = pd.concat([Q_50_BM, df], axis=1)
    Q_70_BM = pd.concat([Q_70_BM, df], axis=1)
    Q_90_BM = pd.concat([Q_90_BM, df], axis=1)

    return Y_r_BM, Q_10_BM, Q_30_BM, Q_50_BM, Q_70_BM, Q_90_BM




def load_dam_data_DS(dam_csv_path):
    # Read data from DAM CSV file.
    date_format = "%d/%m/%Y %H:%M"
    date_parse = lambda date: dt.datetime.strptime(date, date_format)
    dat = pd.read_csv(dam_csv_path)
    dat1 = pd.DataFrame(dat)
    dat1 = dat1.iloc[152:, :].reset_index(drop=True)

    # Create a range of values so the index for the DAM and BM match, ensuring no overlap in trades. 
    # Use them to create a new DataFrame, with P_DAM used to identify DAM values.
    start = 0
    end = 17519
    step = 2
    values = list(range(start, end+1, step))
    df1 = pd.DataFrame({'P_dam': values})

    column_names = ['EURPrices+{}_Forecast_10'.format(i) for i in range(0, 24)]
    Q_10_DAM = dat1[column_names].dropna().stack().reset_index()
    column_names = ['EURPrices+{}_Forecast_30'.format(i) for i in range(0, 24)]
    Q_30_DAM = dat1[column_names].dropna().stack().reset_index()
    column_names = ['EURPrices+{}_Forecast_50'.format(i) for i in range(0, 24)]
    Q_50_DAM = dat1[column_names].dropna().stack().reset_index()
    column_names = ['EURPrices+{}_Forecast_70'.format(i) for i in range(0, 24)]
    Q_70_DAM = dat1[column_names].dropna().stack().reset_index()
    column_names = ['EURPrices+{}_Forecast_90'.format(i) for i in range(0, 24)]
    Q_90_DAM = dat1[column_names].dropna().stack().reset_index()
    column_names = ['EURPrices+{}'.format(i) for i in range(0, 24)]
    Y_r_DAM = dat1[column_names].dropna().stack().reset_index()
    Y_r_DAM = Y_r_DAM.iloc[:, :]

    # Set 'Price' columns for each DataFrame in DAM.
    Y_r_DAM["Price"] = Y_r_DAM.iloc[:, 2:3]
    Q_10_DAM["Price"] = Q_10_DAM.iloc[:, 2:3]
    Q_30_DAM["Price"] = Q_30_DAM.iloc[:, 2:3]
    Q_50_DAM["Price"] = Q_50_DAM.iloc[:, 2:3]
    Q_70_DAM["Price"] = Q_70_DAM.iloc[:, 2:3]
    Q_90_DAM["Price"] = Q_90_DAM.iloc[:, 2:3]

    # Concatenate each DataFrame with the previously created 'df1' DataFrame. 
    # BM and DAM predictions and prices are now held together. Both index values and 'level_0' values align.
    Y_r_DAM = pd.concat([Y_r_DAM, df1], axis=1)
    Q_10_DAM = pd.concat([Q_10_DAM, df1], axis=1)
    Q_30_DAM = pd.concat([Q_30_DAM, df1], axis=1)
    Q_50_DAM = pd.concat([Q_50_DAM, df1], axis=1)
    Q_70_DAM = pd.concat([Q_70_DAM, df1], axis=1)
    Q_90_DAM = pd.concat([Q_90_DAM, df1], axis=1)

    return Y_r_DAM, Q_10_DAM, Q_30_DAM, Q_50_DAM, Q_70_DAM, Q_90_DAM


















# Simulate a charging or discharging operation based on price indexes.
# The bottleneck-controlled strategy is applied to maximize profit while considering battery constraints.
# Adjust charging or discharging dynamically based on the order of minimum and maximum price periods.
def process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index):
    # Check the price indices and execute trading strategies accordingly.
    if min_price_index < max_price_index:
        # Define bottlenecks to control the charging or discharging process.
        bottleneck_1 = min(capacity - charge_level, ramp_rate)
        charge_level += bottleneck_1
        bottleneck_2 = min(charge_level - min_charge_level, ramp_rate)
        charge_level -= bottleneck_2
        # Calculate profit based on trade prices, efficiency, and bottlenecks.
        profit = (current_df.loc[max_price_index // 2, 'Price'] * bottleneck_2 * eff_1) - ((current_df.loc[min_price_index // 2, 'Price'] * bottleneck_1) / eff_2)
        # Append trade details to the 'prices' list.
        prices.append((min_price_index, current_df.loc[min_price_index // 2, 'Price'], max_price_index, current_df.loc[max_price_index // 2, 'Price'], profit, charge_level))
    elif min_price_index > max_price_index:
        # Handle the case when min price index is greater than max price index.
        bottleneck_2 = min(charge_level - min_charge_level, ramp_rate)
        charge_level -= bottleneck_2
        bottleneck_1 = min(capacity - charge_level, ramp_rate)
        charge_level += bottleneck_1
        profit = (current_df.loc[max_price_index // 2, 'Price'] * bottleneck_2 * eff_1) - ((current_df.loc[min_price_index // 2, 'Price'] * bottleneck_1) / eff_2)
        prices.append((min_price_index, current_df.loc[min_price_index // 2, 'Price'], max_price_index, current_df.loc[max_price_index // 2, 'Price'], profit, charge_level))
    return charge_level

# Function to process BM market prices. Similar structure as process_prices_DAM but for BM market.
# Note: In BM market, there is no price index division by 2.
def process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df_bm, min_price_index, max_price_index):
    if min_price_index < max_price_index:
        bottleneck_1 = min(capacity - charge_level, ramp_rate)
        charge_level += bottleneck_1
        bottleneck_2 = min(charge_level - min_charge_level, ramp_rate)
        charge_level -= bottleneck_2
        profit = (current_df_bm.loc[max_price_index, 'Price'] * bottleneck_2 * eff_1) - ((current_df_bm.loc[min_price_index, 'Price'] * bottleneck_1) / eff_2)
        prices.append((min_price_index, current_df_bm.loc[min_price_index, 'Price'], max_price_index, current_df_bm.loc[max_price_index, 'Price'], profit, charge_level))
    elif min_price_index > max_price_index:
        bottleneck_2 = min(charge_level - min_charge_level, ramp_rate)
        charge_level -= bottleneck_2
        bottleneck_1 = min(capacity - charge_level, ramp_rate)
        charge_level += bottleneck_1
        profit = (current_df_bm.loc[max_price_index, 'Price'] * bottleneck_2 * eff_1) - ((current_df_bm.loc[min_price_index, 'Price'] * bottleneck_1) / eff_2)
        prices.append((min_price_index, current_df_bm.loc[min_price_index, 'Price'], max_price_index, current_df_bm.loc[max_price_index, 'Price'], profit, charge_level))
    return charge_level

# Recursive function to explore possible trade pairs within identified price subsets.
# The function iteratively identifies trade pairs within price data and tracks the state of the charge level.
# It considers price data before, in-between, and after each trade pair, maximizing trading opportunities.
def process_recursive_DAM(remaining_prices, charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, level_0, current_Q_A, current_Q_B, Q_A_Preds, Q_B_Preds):
    if len(remaining_prices) > 1:
        max_price_index = remaining_prices['Price'].idxmax()
        min_price_index = remaining_prices['Price'].idxmin()
        # Fetch max and min price indices.
        max_price_index = remaining_prices.loc[max_price_index, 'P_dam']
        min_price_index = remaining_prices.loc[min_price_index, 'P_dam']
        
        # Check and execute DAM trading strategies as appropriate.        
        if current_Q_B.loc[min_price_index / 2, 'Price'] < current_Q_A.loc[max_price_index / 2, 'Price']:
            charge_level = process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)
        else:
            return charge_level  
        # Continue recursively with smaller and larger indices.
        # Prepare data subsets for the next iteration.
        smaller_index = min(min_price_index, max_price_index)
        larger_index = max(min_price_index, max_price_index)
        remaining_prices = current_df[(current_df['P_dam'] > smaller_index) & (current_df['P_dam'] < larger_index)]
        remaining_prices_A = Q_A_Preds[(Q_A_Preds['level_0'] == level_0) & (Q_A_Preds['P_dam'] > smaller_index) & (Q_A_Preds['P_dam'] < larger_index)]
        remaining_prices_B = Q_B_Preds[(Q_B_Preds['level_0'] == level_0) & (Q_B_Preds['P_dam'] > smaller_index) & (Q_B_Preds['P_dam'] < larger_index)]

        charge_level = process_recursive_DAM(remaining_prices, charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, level_0, current_Q_A, current_Q_B, Q_A_Preds, Q_B_Preds)
    
    return charge_level



# Importantly, this code focuses on a dual markets approach, TS3-Dual, which optimizes trading decisions in both DAM and BM markets.
# The DAM positions are determined earlier (t), while BM positions are established later (t+12). BM positions use an 8-hour ahead forecast.
def dual_strat(df, df_bm, Q_A_Preds, Q_B_Preds, Q_A_Preds_bm, Q_B_Preds_bm,  eff_1, eff_2, capacity,charge_level, ramp_rate, min_charge_level):
    prices = [] # Create an empty list to store trading results
    charge_level = charge_level # Initialize the charge level.
    
    # Get unique values of the 'level_0' column in the DAM dataset.
    level_0_values = df['level_0'].unique()

    for level_0 in level_0_values:
        # Filter data for the current 'level_0' value in both DAM and BM datasets.
        current_df = df[df['level_0'] == level_0]
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == level_0]
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == level_0]
        current_df_bm = df_bm[df_bm['level_0'] == level_0]
        current_Q_A_bm = Q_A_Preds_bm[Q_A_Preds_bm['level_0'] == level_0]
        current_Q_B_bm = Q_B_Preds_bm[Q_B_Preds_bm['level_0'] == level_0]  
        
        # Find max and min price indices for DAM.
        max_price_index = current_Q_A['Price'].idxmax()
        min_price_index = current_Q_B['Price'].idxmin()        
        max_price_index = current_Q_A.loc[max_price_index, 'P_dam']
        min_price_index = current_Q_B.loc[min_price_index, 'P_dam']        

        # Determine smaller and larger price indices to create data subsets.
        smaller_index = min(min_price_index, max_price_index)
        larger_index = max(min_price_index, max_price_index)

        # Split data into subsets based on price indices.
        DAM_Intraday = current_df[(current_df['P_dam'] > smaller_index) & (current_df['P_dam'] < larger_index)]
        DAM_Intraday_Q1 = Q_A_Preds[(Q_A_Preds['level_0'] == level_0) & (Q_A_Preds['P_dam'] > smaller_index) & (Q_A_Preds['P_dam'] < larger_index)]
        DAM_Intraday_Q2 = Q_B_Preds[(Q_B_Preds['level_0'] == level_0) & (Q_B_Preds['P_dam'] > smaller_index) & (Q_B_Preds['P_dam'] < larger_index)]
        
        BM_after_DAM = current_df_bm[(current_df_bm.index > smaller_index) & (current_df_bm.index > larger_index)]
        BM_after_DAM_Q1 = Q_A_Preds_bm[(Q_A_Preds_bm['level_0'] == level_0) & (Q_A_Preds_bm.index > smaller_index) & (Q_A_Preds_bm.index > larger_index)]
        BM_after_DAM_Q2 = Q_B_Preds_bm[(Q_B_Preds_bm['level_0'] == level_0) & (Q_B_Preds_bm.index > smaller_index) & (Q_B_Preds_bm.index > larger_index)]
        
        BM_before_DAM = current_df_bm[(current_df_bm.index < smaller_index) & (current_df_bm.index < larger_index)]
        BM_before_DAM_Q1 = Q_A_Preds_bm[(Q_A_Preds_bm['level_0'] == level_0) & (Q_A_Preds_bm.index < smaller_index) & (Q_A_Preds_bm.index < larger_index)]
        BM_before_DAM_Q2 = Q_B_Preds_bm[(Q_B_Preds_bm['level_0'] == level_0) & (Q_B_Preds_bm.index < smaller_index) & (Q_B_Preds_bm.index < larger_index)]
        
        
        #DAM Initial Trade - dictates min max period for BM
        if current_Q_B.loc[min_price_index/2, 'Price'] < current_Q_A.loc[max_price_index/2, 'Price']:
            charge_level = process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)            
        else:
            continue
                                     
        #DAM Trade - optimising inbetween min max period                        
        if len(DAM_Intraday) > 1:
            charge_level = process_recursive_DAM(DAM_Intraday, charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, DAM_Intraday_Q1, DAM_Intraday_Q2, level_0, current_Q_A, current_Q_B, Q_A_Preds, Q_B_Preds)
        else:
            continue

#         if len(BM_before_DAM) > 1:
#             charge_level = process_recursive_bm(BM_before_DAM, charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df_bm, BM_before_DAM_Q1, BM_before_DAM_Q2, level_0, current_Q_A_bm, current_Q_B_bm, Q_A_Preds_bm, Q_B_Preds_bm)
#         else:
#             continue 

        #BM Trade in period before dam trades
        if len(BM_before_DAM) > 1:
                max_price_index_2 = BM_before_DAM_Q1['Price'].idxmax()
                min_price_index_2 = BM_before_DAM_Q2['Price'].idxmin()
                if current_Q_B_bm.loc[min_price_index_2, 'Price'] < current_Q_A_bm.loc[max_price_index_2, 'Price']:
                    charge_level = process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df_bm, min_price_index_2, max_price_index_2)
                else:
                    continue                      

#         if len(BM_after_DAM) > 1:
#             charge_level = process_recursive_bm(BM_after_DAM, charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df_bm, BM_after_DAM_Q1, BM_after_DAM_Q2, level_0, current_Q_A_bm, current_Q_B_bm, Q_A_Preds_bm, Q_B_Preds_bm)
#         else:
#             continue 

        # This section of code deals with trading in the period after the DAM trades.
        # It checks for BM trades in the period after the DAM but before the specified limits.
        if len(BM_after_DAM) > 1:
                max_price_index_1 = BM_after_DAM_Q1['Price'].idxmax()
                min_price_index_1 = BM_after_DAM_Q2['Price'].idxmin()
                if current_Q_B_bm.loc[min_price_index_1, 'Price'] < current_Q_A_bm.loc[max_price_index_1, 'Price']:
                    # If BM trade conditions are met, execute the BM trading process.
                    charge_level = process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df_bm, min_price_index_1, max_price_index_1)
                else:
                    continue     

    # The code iterates through different periods in DAM and BM markets, optimizing trading decisions.
    # It also handles synchronization between DAM and BM markets to enhance trading efficiency.                    
    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice','maxPriceIndex', 'maxPrice', 'profit', 'charge Level'])
    
    
    
def calculate_trading_results_DS(Y_r_BM, Q_10_BM, Q_30_BM, Q_50_BM, Q_70_BM, Q_90_BM, Y_r_DAM, Q_10_DAM, Q_30_DAM, Q_50_DAM, Q_70_DAM, Q_90_DAM):
    eff_1 = 0.8
    eff_2 = 0.98
    

    
    r_dam_bm_50_50=dual_strat(df=Y_r_DAM, df_bm=Y_r_BM, Q_A_Preds=Q_50_DAM, Q_B_Preds=Q_50_DAM, Q_A_Preds_bm=Q_50_BM, Q_B_Preds_bm=Q_50_BM, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=1, ramp_rate=1, min_charge_level=0)
    r_dam_bm_10_30=dual_strat(df=Y_r_DAM, df_bm=Y_r_BM, Q_A_Preds=Q_10_DAM, Q_B_Preds=Q_30_DAM, Q_A_Preds_bm=Q_10_BM, Q_B_Preds_bm=Q_30_BM, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=1, ramp_rate=1, min_charge_level=0)
    r_dam_bm_30_50=dual_strat(df=Y_r_DAM, df_bm=Y_r_BM, Q_A_Preds=Q_30_DAM, Q_B_Preds=Q_50_DAM, Q_A_Preds_bm=Q_30_BM, Q_B_Preds_bm=Q_50_BM, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=1, ramp_rate=1, min_charge_level=0)
    r_dam_bm_50_70=dual_strat(df=Y_r_DAM, df_bm=Y_r_BM, Q_A_Preds=Q_50_DAM, Q_B_Preds=Q_70_DAM, Q_A_Preds_bm=Q_50_BM, Q_B_Preds_bm=Q_70_BM, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=1, ramp_rate=1, min_charge_level=0)
    r_dam_bm_70_90=dual_strat(df=Y_r_DAM, df_bm=Y_r_BM, Q_A_Preds=Q_70_DAM, Q_B_Preds=Q_90_DAM, Q_A_Preds_bm=Q_70_BM, Q_B_Preds_bm=Q_90_BM, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=1, ramp_rate=1, min_charge_level=0)
    r_dam_bm_30_70=dual_strat(df=Y_r_DAM, df_bm=Y_r_BM, Q_A_Preds=Q_30_DAM, Q_B_Preds=Q_70_DAM, Q_A_Preds_bm=Q_30_BM, Q_B_Preds_bm=Q_70_BM, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=1, ramp_rate=1, min_charge_level=0)
    r_dam_bm_10_90=dual_strat(df=Y_r_DAM, df_bm=Y_r_BM, Q_A_Preds=Q_10_DAM, Q_B_Preds=Q_90_DAM, Q_A_Preds_bm=Q_10_BM, Q_B_Preds_bm=Q_90_BM, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=1, ramp_rate=1, min_charge_level=0)
    PF_dam_bm=dual_strat(df=Y_r_DAM, df_bm=Y_r_BM, Q_A_Preds=Y_r_DAM, Q_B_Preds=Y_r_DAM, Q_A_Preds_bm=Y_r_BM, Q_B_Preds_bm=Y_r_BM,          eff_1=0.8, eff_2=0.98, capacity=1,charge_level=1, ramp_rate=1, min_charge_level=0)
    
    results = {
        'r_dam_bm_50_50': np.round(sum(r_dam_bm_50_50.iloc[:, 4:5].values), 2),
        'r_dam_bm_10_30': np.round(sum(r_dam_bm_10_30.iloc[:, 4:5].values), 2),
        'r_dam_bm_30_50': np.round(sum(r_dam_bm_30_50.iloc[:, 4:5].values), 2),
        'r_dam_bm_50_70': np.round(sum(r_dam_bm_50_70.iloc[:, 4:5].values), 2),
        'r_dam_bm_70_90': np.round(sum(r_dam_bm_70_90.iloc[:, 4:5].values), 2),
        'r_dam_bm_30_70': np.round(sum(r_dam_bm_30_70.iloc[:, 4:5].values), 2),
        'r_dam_bm_10_90': np.round(sum(r_dam_bm_10_90.iloc[:, 4:5].values), 2),
        'PF_dam_bm': np.round(sum(PF_dam_bm.iloc[:, 4:5].values), 2)
    }
    
    return results
    
    
    
def print_results_DS(results):
    print("Trading results for different quantile pairs in the DAM-BM dual strategy:")
    for key, value in results.items():
        if key.startswith('r_dam_bm'):
            quantiles = key.split('_')[3:]
            label = f"{quantiles[0]}-{quantiles[1]}"
            print(f"Total sum for trading quantile {label} pair in the DAM-BM dual strategy is: {value}")
        elif key == 'PF_dam_bm':
            print(f"Total sum for the Perfect Forecast pair in the DAM-BM dual strategy is: {value}")

         
  
    
# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_Dual_Strategy_DAM_BM(Y_r_DAM, Y_r_BM, Q_A_Preds, Q_B_Preds, Q_A_Preds_bm, Q_B_Preds_bm, eff_1, eff_2, capacity,charge_level, ramp_rate, min_charge_level):
    # Run electricity strategy for BM
    HF_trade_dam = dual_strat(df=Y_r_DAM, df_bm=Y_r_BM, Q_A_Preds=Q_50_DAM, Q_B_Preds=Q_70_DAM, Q_A_Preds_bm=Q_50_BM, Q_B_Preds_bm=Q_70_BM, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=1, ramp_rate=1, min_charge_level=0)
    # Plot the profit obtained
    plt.figure(figsize=(10, 6))
    plt.plot(HF_trade_dam.iloc[:, 4:5].values, marker='o', linestyle='-', color='y', label='Profit')
    plt.title('Profit Obtained from High Frequency Strategy (TS3-DUAL)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()
       
         
         
         
            
       

