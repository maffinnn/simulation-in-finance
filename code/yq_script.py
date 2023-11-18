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

# Serialize and save an object
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Write in binary mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Load a previously saved object
def load_object(filename):
    with open(filename, 'rb') as inp:  # Read in binary mode
        return pickle.load(inp)

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
    # logger_yq.info(f"Simulated {n_sim} paths for {count} days.")
  
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
    # TODO: Take note of the order of running (run for low hanging fruit first)
    # Individual testing

    @timeit
    def sim_grid_search_heston(hist_windows: list, n_sims: list, models: list, max_sigmas: list):
        n_sims = sorted(n_sims)
        for n_sim in n_sims:
            if ('gbm' in models):
                logger_yq.info("Doing grid search for GBM")
                for hist_window in hist_windows:
                    logger_yq.info(f"Combination is hist_window: {hist_window}, n_sim: {n_sim}")
                    sim_price_period(start_date=cs.INITIAL_PROD_PRICING_DATE, 
                                    end_date=cs.FINAL_PROD_PRICING_DATE, 
                                    hist_window=hist_window, 
                                    n_sim=n_sim,
                                    plot=False, # Hardcoded
                                    max_sigma=0, # Won't be used anyway
                                    model='gbm')
                pass
            
            if ('heston' in models):
                logger_yq.info("Doing grid search for Heston")
                for hist_window, max_sigma in itertools.product(hist_windows, max_sigmas):
                    logger_yq.info(f"Combination is hist_window: {hist_window}, n_sim: {n_sim}")
                    sim_price_period(start_date=cs.INITIAL_PROD_PRICING_DATE, 
                                    end_date=cs.FINAL_PROD_PRICING_DATE, 
                                    hist_window=hist_window, 
                                    n_sim=n_sim,
                                    plot=False, # Hardcoded
                                    max_sigma=max_sigma,
                                    model='heston')
                
       

            
    # hist_windows = [7, 63, 252]
    # n_sims = [10, 100, 1000]
    # model = ['gbm', 'heston']
    # Only for heston: sigma 0.35, 0.5, 10

    # For final grid search
    # Delete useless folders first, clean logs, adjust plot
    # hist_windows = [7, 63, 252]
    # n_sims = [10, 100, 1000]
    # max_sigmas = [0.5, 1.5, 10] # For heston only
    # models = ['gbm','heston']
    
    # For testing only
    hist_windows = [252]
    n_sims = [1]
    max_sigmas = [1.5]
    models = ['heston']
    sim_grid_search_heston(hist_windows=hist_windows,
                    n_sims=n_sims,
                    models=models,
                    max_sigmas=max_sigmas)
    # model_eval.analyse_rmse()
    # TODO:
    #---------------------------
    # run_heston_sim_test_h()
    # plot_a_figure()

    # read_csv_data_chill('lonn_call.csv') # Cannot read an xlsx file converted to csv file improperly
    pass

