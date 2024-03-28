import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt




def electricity_strategy_Multi_Trade_DAM(df, Q_A_Preds, Q_B_Preds, eff_1, eff_2):
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
        
        # Find the maximum price for the day
        max_price_index = current_Q_A['Price'].idxmax()
        # Establish all the remaining prices for the day that fall before the max_price (these will be used to find the min price)
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index)]
        
        # Find the minimum price for the day
        min_price_index = current_Q_B['Price'].idxmin()
        # Establish all the remaining prices for the day that fall after the min_price (these will be used to find the max price)
        prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > min_price_index)]
        
        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        min_price_index1 = None
        max_price_index1 = None
        
        # Identifying buy-sell pairs, ensuring that buy timestamps precede sell timestamps
        if len(prices_before_max) > 0:
            # Find the index of the minimum price within prices_before_max
            min_price_index1 = prices_before_max['Price'].idxmin()
        if len(prices_after_min) > 0:
            # Find the index of the maximum price within prices_after_min
            max_price_index1 = prices_after_min['Price'].idxmax()
            
            
        # Dealing with potential missing values (i.e., if the first min price is index 24, there will be no min price and vice versa)
        if max_price_index is not None and min_price_index1 is not None and max_price_index1 is not None:
            # Compare the profit potential between two buy-sell pairs. Select the pair with the greater difference in predicted prices
            if (current_Q_A.loc[max_price_index, 'Price'] - current_Q_B.loc[min_price_index1, 'Price']) > (current_Q_A.loc[max_price_index1, 'Price'] - current_Q_B.loc[min_price_index, 'Price']): 
                T1_max_price_index = max_price_index
                T1_min_price_index = min_price_index1
            else:
                T1_max_price_index = max_price_index1
                T1_min_price_index = min_price_index
        elif max_price_index is not None and min_price_index1 is not None:
            # If available min_price_index1 is chosen as the minimum in the remaining data, and max_price_index is chosen as the maximum in the remaining data
            T1_max_price_index = max_price_index
            T1_min_price_index = min_price_index1
        elif max_price_index1 is not None and min_price_index is not None:
            # If available min_price_index is chosen as the minimum in the remaining data, and max_price_index1 is chosen as the maximum in the remaining data
            T1_max_price_index = max_price_index1
            T1_min_price_index = min_price_index
        # Calculate profit for the selected buy-sell pair 
        if T1_max_price_index in current_df.index and T1_min_price_index in current_df.index:
            profit = ((current_df.loc[T1_max_price_index, 'Price']) * eff_1) - ((current_df.loc[T1_min_price_index, 'Price']) / eff_2)
            prices.append((T1_min_price_index, current_df.loc[T1_min_price_index, 'Price'], T1_max_price_index, current_df.loc[T1_max_price_index, 'Price'], profit))
      
 

            
        # Same as previous trade, but the trade pair is in the prices that fall before the previous trade (T1)
        # Full charge/discharge so the order doesn't matter                     
        current_df_before_min = current_df[current_df.index < T1_min_price_index]
        current_Q_A_before_min =  Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index < T1_min_price_index)]
        
        # Find the maximum predicted price for the day, subset 1
        max_price_index_before_min = None
        if not current_Q_A_before_min.empty:
            max_price_index_before_min = current_Q_A_before_min['Price'].idxmax()
        # Establish all the remaining prices for that day that fall before the max price (these will be used to find the min price for pair 1, subset 1)
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index_before_min)]
        
        # Find the minimum predicted price for the day, subset 1        
        current_Q_B_before_min =  Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < T1_min_price_index)]
        min_price_index_before_min = None
        # Find the minimum predicted price for the day, subset 1
        if not current_Q_B_before_min.empty:
            min_price_index_before_min = current_Q_B_before_min['Price'].idxmin()
        # Establish all the remaining prices for that day that fall after the min price (these will be used to find the max price for pair 2, subset 1)
        prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > min_price_index_before_min) & (Q_A_Preds.index < T1_min_price_index)]

        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        min_price_index2 = None
        max_price_index2 = None
        
        # Identifying the min price for the remaining prices that fall before the max, subset 1
        if len(prices_before_max) > 0:
            min_price_index2 = prices_before_max['Price'].idxmin()
        # Identifying the max price for the remaining prices that fall after the min, subset 1
        if len(prices_after_min) > 0:
            max_price_index2 = prices_after_min['Price'].idxmax()

        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        T2_max_price_index = None
        T2_min_price_index = None
        
        # Dealing with potential missing values (i.e., if the first min price is index 24, there will be no min price and vice versa)
        if max_price_index_before_min is not None and min_price_index2 is not None and max_price_index2 is not None:
            # All three indices are available; choosing the pair with the greater difference
            if (current_Q_A.loc[max_price_index_before_min, 'Price'] - current_Q_B.loc[min_price_index2, 'Price']) > (current_Q_A.loc[max_price_index2, 'Price'] - current_Q_B.loc[min_price_index_before_min, 'Price']):
                # if max price: max_price_index_before_min & min price: min_price_index2 are greater choose as such
                T2_max_price_index = max_price_index_before_min
                T2_min_price_index = min_price_index2
            else:
                T2_max_price_index = max_price_index2
                T2_min_price_index = min_price_index_before_min
        elif max_price_index_before_min is not None and min_price_index2 is not None:
            T2_max_price_index = max_price_index_before_min
            T2_min_price_index = min_price_index2
        elif max_price_index2 is not None and min_price_index_before_min is not None:
            T2_max_price_index = max_price_index2
            T2_min_price_index = min_price_index_before_min
        # Calculate profit for the selected buy-sell pair, subset 1
        if T2_max_price_index in current_df.index and T2_min_price_index in current_df.index:
            profit = ((current_df.loc[T2_max_price_index, 'Price']) * eff_1) - ((current_df.loc[T2_min_price_index, 'Price']) / eff_2)
            prices.append((T2_min_price_index, current_df.loc[T2_min_price_index, 'Price'], T2_max_price_index, current_df.loc[T2_max_price_index, 'Price'], profit))
 

              
        #same as previous two trades but the trade pair is in the prices that fall after the first trade pair (T1)
        #**full charge/discharge so order doesnt matter         
        current_df_after_T1max = current_df[current_df.index > T1_max_price_index]
        current_Q_A_after_T1max =  Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > T1_max_price_index)]
        # Find the maximum predicted price for the day, subset 2
        max_price_index_after_T1max = None
        if not current_Q_A_after_T1max.empty:
            max_price_index_after_T1max = current_Q_A_after_T1max['Price'].idxmax()
        # Establish all the remaining prices for that day that fall before the max price (these will be used to find the min price for pair 1, subset 2)
        T3_prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index_after_T1max)& (Q_B_Preds.index > T1_max_price_index)]

        # Find the minimum predicted price for the day
        current_Q_B_after_T1max =  Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index > T1_max_price_index)]
        # Find the minimum predicted price for the day, subset 2
        min_price_index_after_T1max = None
        if not current_Q_B_after_T1max.empty:
            min_price_index_after_T1max = current_Q_B_after_T1max['Price'].idxmin()
        # Establish all the remaining prices for that day that fall after the min price (these will be used to find the max price for pair 2, subset 2)
        T3_prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > min_price_index_after_T1max)]

        
        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        min_price_index3 = None
        max_price_index3 = None

        # Identifying the min price for the remaining prices that fall before the max, subset 2
        if len(T3_prices_before_max) > 0:
            min_price_index3 = T3_prices_before_max['Price'].idxmin()
            
        # Identifying the max price for the remaining prices that fall after the min, subset 2
        if len(T3_prices_after_min) > 0:
            max_price_index3 = T3_prices_after_min['Price'].idxmax()

        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        T3_max_price_index = None
        T3_min_price_index = None  
        
        # Dealing with potential missing values (i.e., if the first min price is index 24, there will be no min price and vice versa)
        if max_price_index_after_T1max is not None and min_price_index3 is not None and max_price_index3 is not None:
            # All three indices are available; choosing the pair with the greater difference, subset 2
            if (current_Q_A.loc[max_price_index_after_T1max, 'Price'] - current_Q_B.loc[min_price_index3, 'Price']) > (current_Q_A.loc[max_price_index3, 'Price'] - current_Q_B.loc[min_price_index_after_T1max, 'Price']):
                T3_max_price_index = max_price_index_after_T1max
                T3_min_price_index = min_price_index3
            else:
                T3_max_price_index = max_price_index3
                T3_min_price_index = min_price_index_after_T1max
        elif max_price_index_after_T1max is not None and min_price_index3 is not None:
            T3_max_price_index = max_price_index_after_T1max
            T3_min_price_index = min_price_index3
        elif max_price_index3 is not None and min_price_index_after_T1max is not None:
            T3_max_price_index = max_price_index3
            T3_min_price_index = min_price_index_after_T1max
        # Calculate profit for the selected buy-sell pair, subset 2
        if T3_max_price_index in current_df.index and T3_min_price_index in current_df.index:
            profit = ((current_df.loc[T3_max_price_index, 'Price']) * eff_1) - ((current_df.loc[T3_min_price_index, 'Price']) / eff_2)
            prices.append((T3_min_price_index, current_df.loc[T3_min_price_index, 'Price'], T3_max_price_index, current_df.loc[T3_max_price_index, 'Price'], profit))

            
            
    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice', 'maxPriceIndex', 'maxPrice', 'profit'])
    
    
def electricity_strategy_Multi_Trade_BM(df, Q_A_Preds, Q_B_Preds, eff_1, eff_2):
    # Create an empty list to store trading results
    prices = []
    # Get unique 8 hour period index from the data
    period_index = df['level_0'].unique()

    # Loop through each 8 hour BM period in the dataset.
    for period in period_index:
        # Filter the current periods data
        current_df = df[df['level_0'] == period]
        # Select the predicted prices for quantile A (Q_A) (alpha) for the current 8 hour period
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == period]
        # Select the predicted prices for complementary quantile B (Q_B) (1-alpha) for the current 8 hour period
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == period]
        
        # Find the maximum price for the 8 hour period
        max_price_index = current_Q_A['Price'].idxmax()
        # Establish all the remaining prices for the 8 hour period that fall before the max_price (these will be used to find the min price)
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == period) & (Q_B_Preds.index < max_price_index)]
        
        # Find the minimum price for the 8 hour period
        min_price_index = current_Q_B['Price'].idxmin()
        # Establish all the remaining prices for the 8 hour period that fall after the min_price (these will be used to find the max price)
        prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == period) & (Q_A_Preds.index > min_price_index)]
        
        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        min_price_index1 = None
        max_price_index1 = None
        
        # Identifying buy-sell pairs, ensuring that buy timestamps precede sell timestamps
        if len(prices_before_max) > 0:
            # Find the index of the minimum price within prices_before_max
            min_price_index1 = prices_before_max['Price'].idxmin()
        if len(prices_after_min) > 0:
            # Find the index of the maximum price within prices_after_min
            max_price_index1 = prices_after_min['Price'].idxmax()
            
            
        # Dealing with potential missing values (i.e., if the first min price is index 16, there will be no min price and vice versa)
        if max_price_index is not None and min_price_index1 is not None and max_price_index1 is not None:
            # Compare the profit potential between two buy-sell pairs. Select the pair with the greater difference in predicted prices
            if (current_Q_A.loc[max_price_index, 'Price'] - current_Q_B.loc[min_price_index1, 'Price']) > (current_Q_A.loc[max_price_index1, 'Price'] - current_Q_B.loc[min_price_index, 'Price']): 
                T1_max_price_index = max_price_index
                T1_min_price_index = min_price_index1
            else:
                T1_max_price_index = max_price_index1
                T1_min_price_index = min_price_index
        elif max_price_index is not None and min_price_index1 is not None:
            # If available min_price_index1 is chosen as the minimum in the remaining data, and max_price_index is chosen as the maximum in the remaining data
            T1_max_price_index = max_price_index
            T1_min_price_index = min_price_index1
        elif max_price_index1 is not None and min_price_index is not None:
            # If available min_price_index is chosen as the minimum in the remaining data, and max_price_index1 is chosen as the maximum in the remaining data
            T1_max_price_index = max_price_index1
            T1_min_price_index = min_price_index
        # Calculate profit for the selected buy-sell pair 
        if T1_max_price_index in current_df.index and T1_min_price_index in current_df.index:
            profit = ((current_df.loc[T1_max_price_index, 'Price']) * eff_1) - ((current_df.loc[T1_min_price_index, 'Price']) / eff_2)
            prices.append((T1_min_price_index, current_df.loc[T1_min_price_index, 'Price'], T1_max_price_index, current_df.loc[T1_max_price_index, 'Price'], profit))
      
 

            
        # Same as previous trade, but the trade pair is in the prices that fall before the previous trade (T1)
        # Full charge/discharge so the order doesn't matter                     
        current_df_before_min = current_df[current_df.index < T1_min_price_index]
        current_Q_A_before_min =  Q_A_Preds[(Q_A_Preds['level_0'] == period) & (Q_A_Preds.index < T1_min_price_index)]
        
        # Find the maximum predicted price for the 8 hour period, subset 1
        max_price_index_before_min = None
        if not current_Q_A_before_min.empty:
            max_price_index_before_min = current_Q_A_before_min['Price'].idxmax()
        # Establish all the remaining prices for that 8 hour period that fall before the max price (these will be used to find the min price for pair 1, subset 1)
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == period) & (Q_B_Preds.index < max_price_index_before_min)]
        
        # Find the minimum predicted price for the 8 hour period, subset 1        
        current_Q_B_before_min =  Q_B_Preds[(Q_B_Preds['level_0'] == period) & (Q_B_Preds.index < T1_min_price_index)]
        min_price_index_before_min = None
        # Find the minimum predicted price for the 8 hour period, subset 1
        if not current_Q_B_before_min.empty:
            min_price_index_before_min = current_Q_B_before_min['Price'].idxmin()
        # Establish all the remaining prices for that 8 hour period that fall after the min price (these will be used to find the max price for pair 2, subset 1)
        prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == period) & (Q_A_Preds.index > min_price_index_before_min) & (Q_A_Preds.index < T1_min_price_index)]

        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        min_price_index2 = None
        max_price_index2 = None
        
        # Identifying the min price for the remaining prices that fall before the max, subset 1
        if len(prices_before_max) > 0:
            min_price_index2 = prices_before_max['Price'].idxmin()
        # Identifying the max price for the remaining prices that fall after the min, subset 1
        if len(prices_after_min) > 0:
            max_price_index2 = prices_after_min['Price'].idxmax()

        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        T2_max_price_index = None
        T2_min_price_index = None
        
        # Dealing with potential missing values (i.e., if the first min price is index 16, there will be no min price and vice versa)
        if max_price_index_before_min is not None and min_price_index2 is not None and max_price_index2 is not None:
            # All three indices are available; choosing the pair with the greater difference
            if (current_Q_A.loc[max_price_index_before_min, 'Price'] - current_Q_B.loc[min_price_index2, 'Price']) > (current_Q_A.loc[max_price_index2, 'Price'] - current_Q_B.loc[min_price_index_before_min, 'Price']):
                # if max price: max_price_index_before_min & min price: min_price_index2 are greater choose as such
                T2_max_price_index = max_price_index_before_min
                T2_min_price_index = min_price_index2
            else:
                T2_max_price_index = max_price_index2
                T2_min_price_index = min_price_index_before_min
        elif max_price_index_before_min is not None and min_price_index2 is not None:
            T2_max_price_index = max_price_index_before_min
            T2_min_price_index = min_price_index2
        elif max_price_index2 is not None and min_price_index_before_min is not None:
            T2_max_price_index = max_price_index2
            T2_min_price_index = min_price_index_before_min
        # Calculate profit for the selected buy-sell pair, subset 1
        if T2_max_price_index in current_df.index and T2_min_price_index in current_df.index:
            profit = ((current_df.loc[T2_max_price_index, 'Price']) * eff_1) - ((current_df.loc[T2_min_price_index, 'Price']) / eff_2)
            prices.append((T2_min_price_index, current_df.loc[T2_min_price_index, 'Price'], T2_max_price_index, current_df.loc[T2_max_price_index, 'Price'], profit))
 

              
        #same as previous two trades but the trade pair is in the prices that fall after the first trade pair (T1)
        #**full charge/discharge so order doesnt matter         
        current_df_after_T1max = current_df[current_df.index > T1_max_price_index]
        current_Q_A_after_T1max =  Q_A_Preds[(Q_A_Preds['level_0'] == period) & (Q_A_Preds.index > T1_max_price_index)]
        # Find the maximum predicted price for the 8 hour period, subset 2
        max_price_index_after_T1max = None
        if not current_Q_A_after_T1max.empty:
            max_price_index_after_T1max = current_Q_A_after_T1max['Price'].idxmax()
        # Establish all the remaining prices for that 8 hour period that fall before the max price (these will be used to find the min price for pair 1, subset 2)
        T3_prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == period) & (Q_B_Preds.index < max_price_index_after_T1max)& (Q_B_Preds.index > T1_max_price_index)]

        # Find the minimum predicted price for the 8 hour period
        current_Q_B_after_T1max =  Q_B_Preds[(Q_B_Preds['level_0'] == period) & (Q_B_Preds.index > T1_max_price_index)]
        # Find the minimum predicted price for the 8 hour period, subset 2
        min_price_index_after_T1max = None
        if not current_Q_B_after_T1max.empty:
            min_price_index_after_T1max = current_Q_B_after_T1max['Price'].idxmin()
        # Establish all the remaining prices for that 8 hour period that fall after the min price (these will be used to find the max price for pair 2, subset 2)
        T3_prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == period) & (Q_A_Preds.index > min_price_index_after_T1max)]

        
        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        min_price_index3 = None
        max_price_index3 = None

        # Identifying the min price for the remaining prices that fall before the max, subset 2
        if len(T3_prices_before_max) > 0:
            min_price_index3 = T3_prices_before_max['Price'].idxmin()
            
        # Identifying the max price for the remaining prices that fall after the min, subset 2
        if len(T3_prices_after_min) > 0:
            max_price_index3 = T3_prices_after_min['Price'].idxmax()

        # Initialize variables to store the indices for alternative minimum and maximum predicted prices
        T3_max_price_index = None
        T3_min_price_index = None  
        
        # Dealing with potential missing values (i.e., if the first min price is index 16, there will be no min price and vice versa)
        if max_price_index_after_T1max is not None and min_price_index3 is not None and max_price_index3 is not None:
            # All three indices are available; choosing the pair with the greater difference, subset 2
            if (current_Q_A.loc[max_price_index_after_T1max, 'Price'] - current_Q_B.loc[min_price_index3, 'Price']) > (current_Q_A.loc[max_price_index3, 'Price'] - current_Q_B.loc[min_price_index_after_T1max, 'Price']):
                T3_max_price_index = max_price_index_after_T1max
                T3_min_price_index = min_price_index3
            else:
                T3_max_price_index = max_price_index3
                T3_min_price_index = min_price_index_after_T1max
        elif max_price_index_after_T1max is not None and min_price_index3 is not None:
            T3_max_price_index = max_price_index_after_T1max
            T3_min_price_index = min_price_index3
        elif max_price_index3 is not None and min_price_index_after_T1max is not None:
            T3_max_price_index = max_price_index3
            T3_min_price_index = min_price_index_after_T1max
        # Calculate profit for the selected buy-sell pair, subset 2
        if T3_max_price_index in current_df.index and T3_min_price_index in current_df.index:
            profit = ((current_df.loc[T3_max_price_index, 'Price']) * eff_1) - ((current_df.loc[T3_min_price_index, 'Price']) / eff_2)
            prices.append((T3_min_price_index, current_df.loc[T3_min_price_index, 'Price'], T3_max_price_index, current_df.loc[T3_max_price_index, 'Price'], profit))

            
            
    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice', 'maxPriceIndex', 'maxPrice', 'profit'])

            
def calculate_trading_results_DAM_MT(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_dam_50_50 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2)
    r_dam_10_30 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=eff_1, eff_2=eff_2)
    r_dam_30_50 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2)
    r_dam_50_70 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2)
    r_dam_70_90 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2)
    r_dam_30_70 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2)
    r_dam_10_90 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2)
    
    PF_DAM = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=1, eff_2=1)
    
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
            
            





def calculate_bm_trading_results_MT(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_bm_50_50 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2)
    r_bm_10_30 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=eff_1, eff_2=eff_2)
    r_bm_30_50 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2)
    r_bm_50_70 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2)
    r_bm_70_90 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2)
    r_bm_30_70 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2)
    r_bm_10_90 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2)
    
    PF_BM = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=1, eff_2=1)
    
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
def plot_profit_Multi_trade_BM(Y_r_bm, Q_A_Preds, Q_B_Preds, eff_1, eff_2):
    # Run electricity strategy for BM
    Multi_trade_bm = electricity_strategy_Multi_Trade_BM(df=Y_r_bm, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(Multi_trade_bm.iloc[:, 4:5].values, marker='o', linestyle='-', color='r', label='Profit')
    plt.title('Profit Obtained from Multi-Trade Strategy (TS2-BM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()

# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_Multi_trade_DAM(Y_r, Q_A_Preds, Q_B_Preds, eff_1, eff_2):
    # Run electricity strategy for BM
    Multi_trade_dam = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(Multi_trade_dam.iloc[:, 4:5].values, marker='o', linestyle='-', color='b', label='Profit')
    plt.title('Profit Obtained from Multi-Trade Strategy (TS2-DAM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()

            
            
