import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


# Simulate a charging or discharging operation based on price indexes.
# The bottleneck-controlled strategy is applied to maximize profit while considering battery constraints.
# Adjust charging or discharging dynamically based on the order of minimum and maximum price periods.
def process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index):
    if min_price_index < max_price_index:
        bottleneck_1 = min(capacity - charge_level, ramp_rate)
        charge_level += bottleneck_1
        bottleneck_2 = min(charge_level - min_charge_level, ramp_rate)
        charge_level -= bottleneck_2
        profit = (current_df.loc[max_price_index, 'Price'] * bottleneck_2 * eff_1) - ((current_df.loc[min_price_index, 'Price'] * bottleneck_1) / eff_2)
        prices.append((min_price_index, current_df.loc[min_price_index, 'Price'], max_price_index, current_df.loc[max_price_index, 'Price'], profit, charge_level))
    elif min_price_index > max_price_index:
        bottleneck_2 = min(charge_level - min_charge_level, ramp_rate)
        charge_level -= bottleneck_2
        bottleneck_1 = min(capacity - charge_level, ramp_rate)
        charge_level += bottleneck_1
        profit = (current_df.loc[max_price_index, 'Price'] * bottleneck_2 * eff_1) - ((current_df.loc[min_price_index, 'Price'] * bottleneck_1) / eff_2)
        prices.append((min_price_index, current_df.loc[min_price_index, 'Price'],  max_price_index, current_df.loc[max_price_index, 'Price'], profit, charge_level))        
    return charge_level

# Recursive function to explore possible trade pairs within identified price subsets.
# The function iteratively identifies trade pairs within price data and tracks the state of the charge level.
# It considers price data before, in-between, and after each trade pair, maximizing trading opportunities.
def recursive_process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, current_Q_A, current_Q_B, day):
    if len(remaining_prices_A) <= 1:
        return charge_level

    max_price_index = remaining_prices_A['Price'].idxmax()
    min_price_index = remaining_prices_B['Price'].idxmin()

    if current_Q_B.loc[min_price_index, 'Price'] < current_Q_A.loc[max_price_index, 'Price']:
        charge_level = process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)

    smaller_index = min(min_price_index, max_price_index)
    larger_index = max(min_price_index, max_price_index)

    remaining_prices_A = current_Q_A[(current_Q_A['level_0'] == day) & (current_Q_A.index > smaller_index) & (current_Q_A.index < larger_index)]
    remaining_prices_B = current_Q_B[(current_Q_B['level_0'] == day) & (current_Q_B.index > smaller_index) & (current_Q_B.index < larger_index)]

    charge_level = recursive_process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, current_Q_A, current_Q_B, day)

    return charge_level

# Execute the bottleneck-controlled trading strategy for TS3.
# This strategy aims to maximize profit by considering battery constraints and flexible timestamps.
# It identifies trade pairs within various price subsets and iterates through the available trading opportunities.
def electricity_strategy_DAM_HF(df, Q_A_Preds, Q_B_Preds,  eff_1, eff_2, capacity,charge_level, ramp_rate, min_charge_level):
    # Initialize an empty list to store trade details and set the initial charge level.
    prices = []
    charge_level = charge_level 
    
    # Get unique day indices from the input data frame.
    day_index = df['level_0'].unique()

    # Loop through each day in the dataset.
    for day in day_index:
        # Filter data for the current day.
        current_df = df[df['level_0'] == day]
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == day]
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == day]
        
        # Find the index of the maximum and minimum predicted prices.        
        max_price_index = current_Q_A['Price'].idxmax()
        min_price_index = current_Q_B['Price'].idxmin()
        
        # Determine the smaller and larger price indices.
        smaller_index = min(min_price_index, max_price_index)
        larger_index = max(min_price_index, max_price_index)

        # Split price data into three subsets: before, in-between, and after the trade pair.
        prices_after_T1 = current_df[(current_df.index > smaller_index) & (current_df.index > larger_index)]
        prices_after_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > smaller_index) & (Q_A_Preds.index > larger_index)]
        prices_after_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index > smaller_index) & (Q_B_Preds.index > larger_index)]

        prices_inbetween_T1 = current_df[(current_df.index > smaller_index) & (current_df.index < larger_index)]
        prices_inbetween_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > smaller_index) & (Q_A_Preds.index < larger_index)]
        prices_inbetween_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index > smaller_index) & (Q_B_Preds.index < larger_index)]

        prices_before_T1 = current_df[(current_df.index < smaller_index) & (current_df.index < larger_index)]
        prices_before_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index < smaller_index) & (Q_A_Preds.index < larger_index)]
        prices_before_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < smaller_index) & (Q_B_Preds.index < larger_index)]

        # For each trade pair, execute the process_prices function before considering in-between and after subsets.
        # This part focuses on the trade pair that comes before the identified min and max.   
        if len(prices_before_T1) > 1:
                max_price_index_3 = prices_before_T1_A['Price'].idxmax()
                min_price_index_3 = prices_before_T1_B['Price'].idxmin()
                if current_Q_B.loc[min_price_index_3, 'Price'] < current_Q_A.loc[max_price_index_3, 'Price']:
                    charge_level = process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index_3, max_price_index_3)            
                else:
                    continue     
        # Process the trade pair identified as "T1" and use recursion to explore additional possible trade pairs.
        if current_Q_B.loc[min_price_index, 'Price'] < current_Q_A.loc[max_price_index, 'Price']:
            charge_level = process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)            
        else:
            continue    
                                   
        if len(prices_inbetween_T1) > 1:
            charge_level = recursive_process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, prices_inbetween_T1_A, prices_inbetween_T1_B, current_Q_A, current_Q_B, day)
        else:
            continue     

        # For trade pairs after the identified min and max, execute process_prices.
        # This part does not currently use recursion for simplicity but can be added for more complex scenarios.
        if len(prices_after_T1) > 1:
            max_price_index_1 = prices_after_T1_A['Price'].idxmax()
            min_price_index_1 = prices_after_T1_B['Price'].idxmin()
            charge_level=charge_level
            if current_Q_B.loc[min_price_index_1, 'Price'] < current_Q_A.loc[max_price_index_1, 'Price']:
                charge_level = process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index_1, max_price_index_1)            
            else:
                continue                                
                                    
    # Return the trade details in a DataFrame.                                            
    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice','maxPriceIndex', 'maxPrice', 'profit', 'charge Level'])




def calculate_trading_results_DAM_HF(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_dam_50_50 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_dam_10_30 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_dam_30_50 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_dam_50_70 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_dam_70_90 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_dam_30_70 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_dam_10_90 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)    
    PF_DAM = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    
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






















# Simulate a charging or discharging operation based on price indexes.
# The bottleneck-controlled strategy is applied to maximize profit while considering battery constraints.
# Adjust charging or discharging dynamically based on the order of minimum and maximum price periods.
def process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index):
    if min_price_index < max_price_index:
        bottleneck_1 = min(capacity - charge_level, ramp_rate)
        charge_level += bottleneck_1
        bottleneck_2 = min(charge_level - min_charge_level, ramp_rate)
        charge_level -= bottleneck_2
        profit = (current_df.loc[max_price_index, 'Price'] * bottleneck_2 * eff_1) - ((current_df.loc[min_price_index, 'Price'] * bottleneck_1) / eff_2)
        prices.append((min_price_index, current_df.loc[min_price_index, 'Price'], max_price_index, current_df.loc[max_price_index, 'Price'], profit, charge_level))
    elif min_price_index > max_price_index:
        bottleneck_2 = min(charge_level - min_charge_level, ramp_rate)
        charge_level -= bottleneck_2
        bottleneck_1 = min(capacity - charge_level, ramp_rate)
        charge_level += bottleneck_1
        profit = (current_df.loc[max_price_index, 'Price'] * bottleneck_2 * eff_1) - ((current_df.loc[min_price_index, 'Price'] * bottleneck_1) / eff_2)
        prices.append((min_price_index, current_df.loc[min_price_index, 'Price'],  max_price_index, current_df.loc[max_price_index, 'Price'], profit, charge_level))        
    return charge_level

# Recursive function to explore possible trade pairs within identified price subsets.
# The function iteratively identifies trade pairs within price data and tracks the state of the charge level.
# It considers price data before, in-between, and after each trade pair, maximizing trading opportunities.
def recursive_process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, current_Q_A, current_Q_B, level_0):
    if len(remaining_prices_A) <= 1:
        return charge_level

    max_price_index = remaining_prices_A['Price'].idxmax()
    min_price_index = remaining_prices_B['Price'].idxmin()

    if current_Q_B.loc[min_price_index, 'Price'] < current_Q_A.loc[max_price_index, 'Price']:
        charge_level = process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)

    smaller_index = min(min_price_index, max_price_index)
    larger_index = max(min_price_index, max_price_index)

    remaining_prices_A = current_Q_A[(current_Q_A['level_0'] == level_0) & (current_Q_A.index > smaller_index) & (current_Q_A.index < larger_index)]
    remaining_prices_B = current_Q_B[(current_Q_B['level_0'] == level_0) & (current_Q_B.index > smaller_index) & (current_Q_B.index < larger_index)]

    charge_level = recursive_process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, current_Q_A, current_Q_B, level_0)

    return charge_level

# Execute the bottleneck-controlled trading strategy for TS3.
# This strategy aims to maximize profit by considering battery constraints and flexible timestamps.
# It identifies trade pairs within various price subsets and iterates through the available trading opportunities.
def electricity_strategy_BM_HF(df, Q_A_Preds, Q_B_Preds,  eff_1, eff_2, capacity,charge_level, ramp_rate, min_charge_level):
    # Initialize an empty list to store trade details and set the initial charge level.
    prices = []
    charge_level = charge_level 
    
    # Get unique 8 hour period indices from the input data frame.
    period_index = df['level_0'].unique()

    # Loop through each 8 hour period in the dataset.
    for period in period_index:
        # Filter data for the current 8 hour period.
        current_df = df[df['level_0'] == period]
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == period]
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == period]
        
        # Find the index of the maximum and minimum predicted prices.        
        max_price_index = current_Q_A['Price'].idxmax()
        min_price_index = current_Q_B['Price'].idxmin()
        
        # Determine the smaller and larger price indices.
        smaller_index = min(min_price_index, max_price_index)
        larger_index = max(min_price_index, max_price_index)

        # Split price data into three subsets: before, in-between, and after the trade pair.
        prices_after_T1 = current_df[(current_df.index > smaller_index) & (current_df.index > larger_index)]
        prices_after_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == period) & (Q_A_Preds.index > smaller_index) & (Q_A_Preds.index > larger_index)]
        prices_after_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == period) & (Q_B_Preds.index > smaller_index) & (Q_B_Preds.index > larger_index)]

        prices_inbetween_T1 = current_df[(current_df.index > smaller_index) & (current_df.index < larger_index)]
        prices_inbetween_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == period) & (Q_A_Preds.index > smaller_index) & (Q_A_Preds.index < larger_index)]
        prices_inbetween_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == period) & (Q_B_Preds.index > smaller_index) & (Q_B_Preds.index < larger_index)]

        prices_before_T1 = current_df[(current_df.index < smaller_index) & (current_df.index < larger_index)]
        prices_before_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == period) & (Q_A_Preds.index < smaller_index) & (Q_A_Preds.index < larger_index)]
        prices_before_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == period) & (Q_B_Preds.index < smaller_index) & (Q_B_Preds.index < larger_index)]

        # For each trade pair, execute the process_prices function before considering in-between and after subsets.
        # This part focuses on the trade pair that comes before the identified min and max.   
        if len(prices_before_T1) > 1:
                max_price_index_3 = prices_before_T1_A['Price'].idxmax()
                min_price_index_3 = prices_before_T1_B['Price'].idxmin()
                if current_Q_B.loc[min_price_index_3, 'Price'] < current_Q_A.loc[max_price_index_3, 'Price']:
                    charge_level = process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index_3, max_price_index_3)            
                else:
                    continue     
        # Process the trade pair identified as "T1" and use recursion to explore additional possible trade pairs.
        if current_Q_B.loc[min_price_index, 'Price'] < current_Q_A.loc[max_price_index, 'Price']:
            charge_level = process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)            
        else:
            continue    
                                   
        if len(prices_inbetween_T1) > 1:
            charge_level = recursive_process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, prices_inbetween_T1_A, prices_inbetween_T1_B, current_Q_A, current_Q_B, period)
        else:
            continue     

        # For trade pairs after the identified min and max, execute process_prices.
        # This part does not currently use recursion for simplicity but can be added for more complex scenarios.
        if len(prices_after_T1) > 1:
            max_price_index_1 = prices_after_T1_A['Price'].idxmax()
            min_price_index_1 = prices_after_T1_B['Price'].idxmin()
            charge_level=charge_level
            if current_Q_B.loc[min_price_index_1, 'Price'] < current_Q_A.loc[max_price_index_1, 'Price']:
                charge_level = process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index_1, max_price_index_1)            
            else:
                continue                                
                                    
    # Return the trade details in a DataFrame.                                            
    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice','maxPriceIndex', 'maxPrice', 'profit', 'charge Level'])



def calculate_bm_trading_results_HF(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_bm_50_50 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_bm_10_30 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_bm_30_50 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_bm_50_70 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_bm_70_90 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_bm_30_70 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    r_bm_10_90 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)    
    PF_BM = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    
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



# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_High_Frequency_Strategy_BM(Y_r_bm, Q_A_Preds, Q_B_Preds, eff_1, eff_2, capacity,charge_level, ramp_rate, min_charge_level):
    # Run electricity strategy for BM
    HF_trade_bm = electricity_strategy_BM_HF(df=Y_r_bm, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(HF_trade_bm.iloc[:, 4:5].values, marker='o', linestyle='-', color='r', label='Profit')
    plt.title('Profit Obtained from High Frequency Strategy (TS3-BM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()

# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_High_Frequency_Strategy_DAM(Y_r, Q_A_Preds, Q_B_Preds, eff_1, eff_2, capacity,charge_level, ramp_rate, min_charge_level):
    # Run electricity strategy for BM
    HF_trade_dam = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(HF_trade_dam.iloc[:, 4:5].values, marker='o', linestyle='-', color='b', label='Profit')
    plt.title('Profit Obtained from High Frequency Strategy (TS3-DAM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    
    


