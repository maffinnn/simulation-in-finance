import typing
import logging
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
from yq.scripts import heston
from yq.utils import option
from yq.utils import calendar
from yq.scripts import simulation as sm
from yq.utils import path as yq_path
from yq.utils import log
from sc import constants as cs
from sc import payoff as po
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

def run_heston_sim_test_h():
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

def plot_a_figure():
    # Copy the start_time_str from the folders
    product_est_date_sim_data_df_list = sm.read_sim_data(
            model_name='gbm',
            start_time_str='20231111_195045_022812', 
            prod_est_start_date=pd.Timestamp('2023-08-09'), 
            prod_est_end_date=pd.Timestamp('2023-11-09'))
    # print(type(product_est_date_sim_data_df_list)[0])
    n_sim_on_day = pd.concat(product_est_date_sim_data_df_list[20], axis=1)
    ax = n_sim_on_day.plot(alpha=0.6, legend=False)
    # Set the title
    ax.set_title("Product Pricing Date: x, Model: Multi Asset GBM")

    # Add text labels for additional features
    ax.text(0.5, 0.95, "n_sim = 100, LONZA, SIKA", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    figure = ax.get_figure()
    figure.savefig(yq_path.get_plots_path(cur_dir).joinpath('sample_sim_path_on_prod_date.png'))

    # print(product_est_date_sim_data_df_list[0])
    logger_yq.info("Testing the logs: %s", product_est_date_sim_data_df_list[0])


def read_csv_data_chill(file_name: str) -> pd.DataFrame:
    curr_dir = Path(__file__).parent.parent
    print(curr_dir)

    file_path = curr_dir.joinpath('data', 'options-test', '20230814', file_name)
    print(file_path)

    options_data = pd.read_csv(file_path)

    return options_data

def plot_graph():
    paths_arr = sm.read_sim_data('heston', '20231113_013012_363749', pd.Timestamp('2023-08-09'), pd.Timestamp('2023-08-09'))
    df_sim = paths_arr[0][0]

    fig, ax = plt.subplots(figsize=(10,6))

    hist_data = po.get_historical_assets_all()
    hist_df = hist_data[(hist_data.index >= cs.INITIAL_FIXING_DATE) 
                            & (hist_data.index <= cs.FINAL_FIXING_DATE)]
    for asset in cs.ASSET_NAMES:
        ax.plot(hist_df.index, hist_df[asset], alpha=0.5, label=asset)
    for col in df_sim.columns:
        ax.plot(df_sim.index, df_sim[col], alpha=0.5, label=col)


    title_str = f"PPD: "
    plt.title(title_str)
    plt.legend(loc='upper right')
    plt.tight_layout()
    stor_dir = yq_path.get_plots_path(Path(__file__).parent)                     
    stor_dir.mkdir(parents=True, exist_ok=True)
    file_path = stor_dir.joinpath(f'test1.png')
    plt.savefig(file_path, bbox_inches='tight')

if __name__ == "__main__":
    # cur_dir = Path(os.getcwd()).parent # ipynb cannot use __file__
    cur_dir = Path(__file__).parent
    logger_yq = log.setup_logger('yq', yq_path.get_logs_path(cur_dir=cur_dir).joinpath(f"log_file_{datetime.datetime.now().strftime('%Y%m%d_%H')}.log"))
    # logger_yq = logging.getLogger('yq')
    # option.format_file_names('options-complete')
    # option.clean_options_data('options-complete')
    # plot_graph()

    #################################################

    tcal = calendar.SIXTradingCalendar()
    # print(bus_date_range)
    # print(bus_date_range.index.to_list())
    start_time_acc = datetime.datetime.now()
    
    # TODO: BEFORE RUNNING: Change the dates, h_array, 
    start_time = time.time()
    count = 0
    for prod_date in tcal.create_six_trading_dates('2023-08-09', '2023-08-09').index:
        try: 
            logger_yq.info(f"Pricing the product on {prod_date}")
            params = {
                    'prod_date': prod_date,
                    'hist_window': 252,
                    'h_array': [[0], [0]],
                    'start_time_acc': start_time_acc,
                    'plot': True
            }
            hst = heston.multi_heston(params)
            # logger_yq.info(f"Heston hist attributes are {vars(heston)}")
            hst.sim_n_path(n_sim=1)
            del hst
            count += 1
        except Exception as e:
            logger_yq.error(f"Error during calibration on {prod_date}: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    min, sec = divmod(elapsed_time, 60)
    logger_yq.info(f"The elapsed time for {count} pricing dates is {int(min)} minutes, {int(sec)} seconds")
    
    #---------------------------
    # run_heston_sim_test_h()
    # plot_a_figure()

    # read_csv_data_chill('lonn_call.csv') # Cannot read an xlsx file converted to csv file improperly

    # Set up specific loggers
   
    # Your plotting code remains the same
    # ...

    # Logging the DataFrame
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