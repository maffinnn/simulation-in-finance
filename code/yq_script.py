import itertools
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
from yq.utils.time import timeit
from yq.utils import io
from yq.scripts import models, model_eval
from yq.scripts import heston, gbm
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


def read_csv_data_chill(file_name: str) -> pd.DataFrame:
    curr_dir = Path(__file__).parent.parent
    print(curr_dir)

    file_path = curr_dir.joinpath('data', 'options-test', '20230814', file_name)
    print(file_path)

    options_data = pd.read_csv(file_path)

    return options_data

@timeit
def plot_graph(model: str, prod_date: str):
    paths_arr = sm.read_sim_data(model, '20231113_185603_63_0.5', pd.Timestamp(prod_date), pd.Timestamp(prod_date))
    n_sim = len(paths_arr[0])
    n_ppd = len(paths_arr)
    logger_yq.info(f"The number of PPD and sims is {n_ppd}, {n_sim} ")
    sim_paths = pd.concat(paths_arr[0], axis=1)
    
    fig, ax = plt.subplots(figsize=(10,6))
    sim_paths.plot()
    
    title_str = f"PPD: {prod_date}"
    plt.title(title_str)
    plt.legend(loc='upper right')
    plt.tight_layout()
    stor_dir = yq_path.get_plots_path(Path(__file__).parent).joinpath(f'{model}',
                                                                          'n_sim')                     
    stor_dir.mkdir(parents=True, exist_ok=True)
    file_path = stor_dir.joinpath(f"{prod_date.strftime('%Y%m%d')}_{n_sim}.png")
    plt.savefig(file_path, bbox_inches='tight')

@timeit
def sim_price_period(start_date: pd.Timestamp, 
                     end_date: pd.Timestamp, 
                     hist_window: int,
                     n_sim: int, 
                     plot: bool,
                     max_sigma: float,
                     model: str):
    tcal = calendar.SIXTradingCalendar()

    start_time_acc = datetime.datetime.now()
    # TODO: BEFORE RUNNING: Change the dates, h_array, 
    count = 0
    for prod_date in tcal.create_six_trading_dates(start_date, end_date).index:
        try: 
            logger_yq.info(f"Pricing the product on {prod_date}")
            params = {
                    'model_name': model,
                    'prod_date': prod_date,
                    'hist_window': hist_window,
                    'h_array': [[0], [0]],
                    'start_time_acc': start_time_acc,
                    'plot': plot,
                    'max_sigma': max_sigma
                }
            if (model == 'heston'):
                
                hst = heston.MultiHeston(params)
                # logger_yq.info(f"Heston hist attributes are {vars(heston)}")
                hst.sim_n_path(n_sim=n_sim)
                del hst
            elif (model == 'gbm'):
                gbm_pricer = gbm.MultiGBM(params)
                gbm_pricer.sim_n_path(n_sim=n_sim)
                del gbm_pricer
            count += 1
        except Exception as e:
            logger_yq.error(f"Error during simulation on {prod_date}: {e}")
    logger_yq.info(f"Simulated {n_sim} paths for {count} days.")
  
if __name__ == "__main__":
    # cur_dir = Path(os.getcwd()).parent # ipynb cannot use __file__
    cur_dir = Path(__file__).parent
    logger_yq = log.setup_logger('yq', yq_path.get_logs_path(cur_dir=cur_dir).joinpath(f"log_file_{datetime.datetime.now().strftime('%Y%m%d_%H')}.log"))
    logger_yq.info("\n##########START##########\n")
    # logger_yq = logging.getLogger('yq')
    # option.format_file_names('options-complete')
    # option.clean_options_data('options-complete')
    # plot_graph()

    #############################################
    # ANALYSIS FUNCTIONS
    # model_eval.analyse_V_t()
    #################################################
    # TODO: Change the acc start time to fix the issues
    # Individual testing

    @timeit
    def sim_grid_search_heston(hist_windows: list, n_sims: list, models: list, max_sigmas: list):
        if ('gbm' in models):
            logger_yq.info("Doing grid search for GBM")
            for hist_window, n_sim in itertools.product(hist_windows, n_sims):
                logger_yq.info(f"Combination is hist_window: {hist_window}, n_sim: {n_sim}")
                sim_price_period(start_date=cs.INITIAL_PROD_PRICING_DATE, 
                                end_date=cs.FINAL_PROD_PRICING_DATE, 
                                hist_window=hist_window, 
                                n_sim=n_sim,
                                plot=True, # Hardcoded
                                max_sigma=0, # Won't be used anyway
                                model='gbm')
            pass
         
        if ('heston' in models):
            logger_yq.info("Doing grid search for Heston")
            for hist_window, n_sim, max_sigma in itertools.product(hist_windows, n_sims, max_sigmas):
                logger_yq.info(f"Combination is hist_window: {hist_window}, n_sim: {n_sim}")
                sim_price_period(start_date=cs.INITIAL_PROD_PRICING_DATE, 
                                end_date=cs.FINAL_PROD_PRICING_DATE, 
                                hist_window=hist_window, 
                                n_sim=n_sim,
                                plot=True, # Hardcoded
                                max_sigma=max_sigma,
                                model='heston')
                
       

            
    # hist_windows = [7, 63, 252]
    # n_sims = [10, 100, 1000]
    # model = ['gbm', 'heston']
    # Only for heston: sigma 0.35, 0.5, 10

    # Test for HESTON MAX SIGMA
    hist_windows = [63]
    n_sims = [3]
    models = ['gbm','heston']
    max_sigmas = [0.5, 1.5, 10]
    sim_grid_search_heston(hist_windows=hist_windows,
                    n_sims=n_sims,
                    models=models,
                    max_sigmas=max_sigmas)
     

    # TODO:
    #---------------------------
    # run_heston_sim_test_h()
    # plot_a_figure()

    # read_csv_data_chill('lonn_call.csv') # Cannot read an xlsx file converted to csv file improperly
    pass

