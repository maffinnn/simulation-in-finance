import typing
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import QuantLib as ql
import yfinance as yf
import pandas_market_calendars as mcal
import datetime
import time
from pathlib import Path
import os

# Self-written modules
from yq.scripts import models
from yq.scripts import heston_func
from yq.utils import option
from yq.utils import calendar
from yq.scripts import simulation as sm
from sc import constants as cs
from sy.variance_reduction import apply_control_variates
from sy.interest_rate import populate_bond_table, get_period
from sy.calibration import apply_empirical_martingale_correction

# data = option.read_options_data("lonn_call.csv")
# print(data)

# # Create an instance of the SIXTradingCalendar class
# trading_calendar = calendar.SIXTradingCalendar()

# # Perform various operations using the class instance
# # For example, calculate the number of business days between two dates
# start_date = pd.Timestamp('2023-01-01')
# end_date = pd.Timestamp('2023-01-10')
# business_days = trading_calendar.calculate_business_days(start_date, end_date)
# print(f"Number of business days: {business_days}")

# Serialize and save an object
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Write in binary mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Load a previously saved object
def load_object(filename):
    with open(filename, 'rb') as inp:  # Read in binary mode
        return pickle.load(inp)


if __name__ == "__main__":
    historical_start_date = '2022-08-09'
    # Define the ticker list
    ticker_list = ['LONN.SW', 'SIKA.SW']

    # Fetch the data
    data = yf.download(ticker_list, historical_start_date)['Adj Close'] # Auto adjust is false
    data.plot()
    
    # model_instance = models.PricingModel(params=params)
    # save_object(model_instance, 'model_instance.pkl')
    # loaded_model_instance = load_object('model_instance.pkl')
    # print(loaded_model_instance.data)

    start_time_acc = datetime.datetime.now() # Track the nth attempt
    print(start_time_acc)
    n_sim = 1
    trading_calendar = calendar.SIXTradingCalendar()
    bus_date_range = trading_calendar.create_six_trading_dates('2023-08-09', '2023-08-09')
    for prod_date in bus_date_range.index:
        params = {
            'data': data,
            'ticker_list': ['LONN.SW', 'SIKA.SW'],
            'prod_date': prod_date
        }
        try:
            start_time = time.time()
            heston = models.PricingModel(params=params)

            sim_data_df = []
            for sim in range(n_sim):
                sim_start_date = trading_calendar.add_trading_day(prod_date, 1)
                
                
                sim_data = heston.multi_asset_heston_model(
                    sim_start_date=sim_start_date, 
                    hist_window=252, 
                    sim_window=trading_calendar.calculate_business_days(sim_start_date, cs.FINAL_FIXING_DATE), 
                    h_adjustment=[0, 0])
                sim_data_df.append(sim_data)
                
                sim_data_h = heston.multi_asset_heston_model(
                    sim_start_date=sim_start_date, 
                    hist_window=252, 
                    sim_window=trading_calendar.calculate_business_days(sim_start_date, cs.FINAL_FIXING_DATE), 
                    h_adjustment=[1, 0])

            end_time = time.time()
            elapsed_time = end_time - start_time
            min, sec = divmod(elapsed_time, 60)
            print(f"The elapsed time is for {n_sim} is {int(min)} minutes, {int(sec)} seconds")
            
            S_T_1 = sim_data.loc[cs.FINAL_FIXING_DATE, 'LONN.SW']
            S_T_2 = sim_data_h.loc[cs.FINAL_FIXING_DATE,'LONN.SW']
            print(S_T_1 / S_T_2)
            
            S_0_1 = data.loc[prod_date, 'LONN.SW']
            S_0_2 = data.loc[prod_date, 'LONN.SW'] + 1
            print(S_0_1 / S_0_2)
            print(S_T_1, S_T_2, S_0_1, S_0_2)
        except Exception as e:
            # Log the error with the date that caused it
            raise Exception("MultiHeston has error.")
        
    pass
    # calendar = calendar.SIXTradingCalendar()
    # dates = calendar.create_six_trading_dates('2023-08-09', '2023-11-09')
    # for date in dates.index:
    #     option.create_csv_files(date)
    # print("CSV files created")

    # params = {
    # 'data': data,
    # 'ticker_list': ['LONN.SW', 'SIKA.SW']
    # }
    # trading_calendar = calendar.SIXTradingCalendar()
    # heston = models.PricingModel(params = params)

    # bus_date_range = trading_calendar.create_six_trading_dates('2023-08-09', '2023-08-09')
    # # print(bus_date_range)
    # # print(bus_date_range.index.to_list())
    # for product_est_date in bus_date_range.index:
    #     # print(date, type(date))
        
    #     try:
    #         sim_start_date = trading_calendar.add_trading_day(product_est_date, 1)
    #         sim_data = heston.multi_asset_heston_model(
    #             sim_start_date=sim_start_date, 
    #             hist_window=trading_calendar.calculate_business_days(sim_start_date, 
    #                                                                 cs.FINAL_FIXING_DATE), 
    #             sim_window=252, h_adjustment=[0, 0])
    #         # print(sim_data)
    #     except Exception as e:
    #         # Log the error with the date that caused it
    #         raise Exception("Heston has error.")
        
    #     # sim_data.columns = ['LONN.SW', 'SIKA.SW']
    #     # sim_data['LONN.SW'] 