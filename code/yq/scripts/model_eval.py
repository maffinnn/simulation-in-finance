import logging
import re
import pandas as pd
import typing
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from yq.utils import io
from yq.utils.time import timeit
from yq.scripts import heston_func as hf
from yq.scripts import simulation as sm
from yq.utils import option, calendar, log, path as yq_path
from sc import constants as cs
from sc import payoff as po
from sy.interest_rate import populate_bond_table
import datetime

pd.set_option('display.max_rows', None)  # Set to None to display all rows
logger_yq = logging.getLogger('yq')

def analyse_V_t():
    # Sample data from the user input
    curr_dir = Path(__file__)
    file_path = yq_path.get_root_dir(cur_dir=curr_dir).joinpath('code', 'yq', 'inter-data', '20230809_max_sigma_1.5_v_t.txt')

    # Read the contents of the file
    with open(file_path, 'r') as file:
        data = file.read()

    # Regular expression to capture the required data
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - yq - INFO - The V_t value for \d+th iteration asset (\d) is: ([\d.]+)"

    # Finding all matches
    matches = re.findall(pattern, data)
    # logger_yq.info(f"Mathes is\n {matches}")

    # Append a dict for each row in df
    data_list = []
    for match in matches:
        asset, value = match
        if asset == "0":
            data_list.append({"V_t_LONN.SE": value})
        elif asset == "1":
            data_list[-1]["V_t_SIKA.SE"] = value

    # Creating DataFrame
    df = pd.DataFrame(data_list)
    logger_yq.info(f"Dataframe for V_t data is\n{df}")

    df['V_t_LONN.SE'] = pd.to_numeric(df['V_t_LONN.SE'])
    df['V_t_SIKA.SE'] = pd.to_numeric(df['V_t_SIKA.SE'])

    logger_yq.info(f"Dataframe info is\n{df.describe()}")

    plot_dir = yq_path.get_plots_path(cur_dir=curr_dir)
    stor_dir = plot_dir.joinpath('options')
    stor_dir.mkdir(parents=True, exist_ok=True)
    file_path = stor_dir.joinpath('v_t_over_time.png')
    # Data Visualization
    df.plot(kind='line')
    plt.title('V_t Values Over Time\nDate: 2023-08-09, max_sigma = 1.5, kappa = [3.70, 8.30]')
    plt.xlabel('Simulation time steps')
    plt.ylabel('V_t Values')
    plt.axhline(0.02714370413916297, color='r', linestyle='-.', label='Theta for LONN.SE')
    plt.axhline(0.012215540619082728, color='g', linestyle='-.', label='Theta for SIKA.SE')
    plt.savefig(file_path)

def analyse_rho_volatility():
    pass
    # Plot rho and the volatility
    # 1st day is 

    # TODO: yq


def analyse_volatility():
    data = po.get_historical_assets_all()
    prod_date = pd.Timestamp('2023-08-09')
    S_0_vector = [data.loc[prod_date, asset] for asset in cs.ASSET_NAMES]

    
    lonn_call = option.read_clean_options_data(options_dir='options-cleaned', curr_date=prod_date, file_name="lonn_call.csv")
    sika_call = option.read_clean_options_data(options_dir='options-cleaned', curr_date=prod_date, file_name="sika_call.csv")

    for index, df in enumerate([lonn_call, sika_call]):
        df = df[['maturity', 'IV', 'strike']]
        df['moneyness'] = S_0_vector[index] / df['strike']

        logger_yq.info(f"Dataframe of volatility smile is\n{df.head()}")
    
    pass
    # Fetch data form options

def extract_file_name_gbm(input_string):
    # Split the string using underscore as the delimiter
    parts = input_string.split('_')

    # Extract the date_time part
    date_time = parts[0] + '_' + parts[1]

    # Extract the last digit of the string
    last_digit = parts[2]

    return date_time, last_digit



@timeit
# hist_windows = [63]
#     n_sims = [3]
#     models = ['gbm','heston']
#     max_sigmas = [0.5, 1.5, 10]
# n_sim can be calculated
# hist_windows: list, n_sims: list, models: list, max_sigmas: list

# Roughly 30 sec to read the 1K file
def analyse_rmse(model: str):
    # TODO: Take the values from yq_script
    gbm_files = ["20231114_024525_7", "20231114_024646_63", "20231114_024808_252", 
                 "20231114_030106_7", "20231114_031302_63", "20231114_032613_252",
                 "20231114_052051_7", "20231114_072227_63", "20231114_092701_252"]
    
    # gbm_files = ["20231114_024525_7", "20231114_030106_7", "20231114_031302_63", "20231114_032613_252"]
    RMSE_dict = {}
    for uid in gbm_files: # TODO: Change
        print(uid)
        # Getting back the strings
        if model == 'gbm':
            time_str, hist_wdw = extract_file_name_gbm(uid)
            print(time_str, hist_wdw)
        else:
            pass

        paths_arr, dates = sm.read_sim_data(model, uid, cs.INITIAL_PROD_PRICING_DATE, cs.FINAL_PROD_PRICING_DATE)
        n_sim = len(paths_arr[0])
        n_ppd = len(paths_arr) # Some missing lists in n_ppd
        print(n_sim, n_ppd)

        actual_price = po.get_product_price(cs.FINAL_PROD_PRICING_DATE).rename(columns={'Price': 'actual_price'})
        # Actual price for the product price period
        actual_price = actual_price[actual_price.index >= cs.INITIAL_PROD_PRICING_DATE]
        # logger_yq.info(f'Actual_price df is\n{actual_price}')

        payouts_compare = pd.DataFrame({'ppd_payouts': np.zeros(len(actual_price))})
        payouts_compare.index = pd.to_datetime(actual_price.index)
        payouts_compare = pd.concat([payouts_compare, actual_price], axis=1)
        
        # logger_yq.info("The payouts compare df is {payouts_compare}")

        # Average payouts of all the sim paths on each ppd
        for ppd in range(n_ppd):
            # Need to rename columns first
            # logger_yq.info(f'pdd = {ppd} paths_arr[ppd] is\n{paths_arr[ppd]}')
            # Payouts for all the paths on one day of the price period (multiple paths)
            if (len(paths_arr[ppd]) != 0):
                paths_payout = po.pricing_multiple(paths_arr[ppd])
                payouts_compare.loc[pd.Timestamp(dates[ppd]),'ppd_payouts'] =  np.mean(paths_payout)
            else:
                logger_yq.error(f"Path payouts cannot be 0")
        # print(payouts_compare)

        # For Heston need to deal with empty PPD, calculate 1 more RMSE_clean
        if model == 'heston':
            compare_clean = payouts_compare.copy(deep=True)
            compare_clean = compare_clean[(compare_clean.index < pd.Timestamp('2023-09-15')) | (compare_clean.index > pd.Timestamp('2023-10-01'))]
            compare_clean = compare_clean[compare_clean['ppd_payouts'] > 0]
            RMSE_clean = np.sqrt(np.mean((compare_clean['ppd_payouts'] - compare_clean['actual_price']) ** 2))

        RMSE = np.sqrt(np.mean((payouts_compare['ppd_payouts'] - payouts_compare['actual_price']) ** 2))

        
        if model == 'heston':
            # RMSE_dict[""] = RMSE
            # RMSE_dict[] = RMSE_clean

            # logger_yq.info(f"RMSE for combination is:\n{RMSE} {RMSE_clean}") # TODO: Add combination
            # print(RMSE, RMSE_clean)
            title_str = f"model, hist_wdw, max_sigma"
            pass
        else:
            RMSE_dict[f"{model}_{uid}_{hist_wdw}_{n_sim}_{n_ppd}"] = RMSE
            # logger_yq.info(f"RMSE for combination is:\n{RMSE} {RMSE_clean}") # TODO: Add combination
            print(RMSE)

            fig, ax = plt.subplots(figsize=(10, 6))
            plt.tight_layout()
            title_str = f"Model: {model}, hist_wdw: {hist_wdw}"
            subtitle_str = f"n_sim: {n_sim}, n_ppd: {n_ppd}, RMSE: {RMSE:.2f}"
            plt.title(f"{title_str}\n{subtitle_str}")  # Adjust font size as needed
            
            
            ax.plot(payouts_compare.index, payouts_compare['ppd_payouts'],
                    alpha = 1, label='ppd_payouts') # Must use this to see title
            ax.plot(payouts_compare.index, payouts_compare['actual_price'],
                    alpha = 1, label='actual_price')
            plt.legend(loc='upper right')

            stor_dir = yq_path.get_plots_path(Path(__file__).parent).joinpath('eval', model)
            file_path = stor_dir.joinpath(f'{model}_{uid}_{n_sim}_{n_ppd}.png')
            stor_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

    print(RMSE_dict)


        

   



