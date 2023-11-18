import logging
import re
import pandas as pd
import typing
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    plt.legend()
    plt.savefig(file_path)

def analyse_rho_volatility():
    pass
    # Plot rho and the volatility
    # 1st day is 

    # TODO: yq

def anal_vol():
    data = po.get_historical_assets_all()
    prod_date = pd.Timestamp('2023-08-09')
    S_0_vector = [data.loc[prod_date, asset] for asset in cs.ASSET_NAMES]

    
    lonn_call = option.read_clean_options_data(options_dir='options-cleaned', curr_date=prod_date, file_name="lonn_call.csv")
    sika_call = option.read_clean_options_data(options_dir='options-cleaned', curr_date=prod_date, file_name="sika_call.csv")

    df_list = [lonn_call, sika_call]
    for index, df in enumerate(df_list):
        df = df[['maturity', 'IV', 'strike']]
        df['moneyness'] = S_0_vector[index] / df['strike']

        logger_yq.info(f"Dataframe of volatility smile is\n{df.head()}")

    # plot_volatility(df=df, title)

def plot_volatility(df, title):

    # Unique maturities for different plots
    maturities = df['maturity'].unique()

    plt.figure(figsize=(12, 6))

    for mat in maturities:
        subset = df[df['maturity'] == mat]
        # Scatter plot
        plt.scatter(subset['moneyness'], subset['IV'], label=f'Maturity: {mat:.2f}')

        # # Apply LOWESS smoothing
        # lowess = statsm.nonparametric.lowess(subset['IV'], subset['moneyness'], frac=0.1)

        # # Plot the smoothed line
        # plt.plot(lowess[:, 0], lowess[:, 1], label=f'Smoothed {mat:.2f}')

    plt.title('2D Plot of IV vs. Moneyness for LONN.SE on 2023-08-09')
    plt.xlabel('Moneyness')
    plt.ylabel('Implied Volatility (IV)')
    plt.legend()
    plt.show()
    plt.close()


    # 3D Scatter Plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['moneyness'], df['maturity'], df['IV'], c='r', marker='o')
    ax.set_title(f'3D Scatter Plot of IV for {title}')
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility (IV)')
    plt.show()
    plt.close()

    pass
    # Fetch data form options

def extract_file_name_gbm(input_string):
    # Split the string using underscore as the delimiter
    parts = input_string.split('_')
    date_time = parts[0] + '_' + parts[1]
    hist_wdw = parts[2]

    return date_time, hist_wdw

def extract_file_name_heston(input_string):
    # Split the string using underscore as the delimiter
    parts = input_string.split('_')
    date_time = parts[0] + '_' + parts[1]
    hist_wdw = parts[2]
    max_sigma = parts[3]


    return date_time, hist_wdw, max_sigma

@timeit
# hist_windows = [63]
#     n_sims = [3]
#     models = ['gbm','heston']
#     max_sigmas = [0.5, 1.5, 10]
# n_sim can be calculated
# hist_windows: list, n_sims: list, models: list, max_sigmas: list

# Roughly 30 sec to read the 1K file
def analyse_rmse(model: str):
    # TODO: Add heston params
    # gbm_files = ["20231114_024525_7", "20231114_024646_63", "20231114_024808_252", 
    #              "20231114_030106_7", "20231114_031302_63", "20231114_032613_252",
    #              "20231114_052051_7", "20231114_072227_63", "20231114_092701_252"]
    # heston_files = ["20231114_024931_7_0.5", "20231114_025048_7_1.5", "20231114_025152_7_10",
    #                 "20231114_025252_63_0.5", "20231114_025415_63_1.5", "20231114_025539_63_10",
    #                 "20231114_025704_252_0.5", "20231114_025829_252_1.5", "20231114_025950_252_10",
    #                 "20231114_033852_7_0.5", "20231114_034926_7_1.5", "20231114_035807_7_10",
    #                 "20231114_040708_63_0.5", "20231114_041953_63_1.5", "20231114_043237_63_10",
    #                 "20231114_044509_252_0.5", "20231114_045744_252_1.5", "20231114_050915_252_10",   
    #                 "20231114_115230_7_0.5", "20231114_134109_7_1.5", "20231114_151836_7_10",
    #                 "20231114_165326_63_0.5", "20231114_191500_63_1.5"]
    # heston_files = ["20231114_220543_63_10", 
    #                 "20231115_010234_252_0.5", "20231115_032413_252_1.5", "20231115_051421_252_10"]
    # gbm_files = ["20231114_092701_252"]
    heston_files = ['20231118_211401_252_1.5']
    RMSE_dict = {}
    for uid in heston_files: # TODO: Change
        print(uid)
        # Getting back the strings, rmb to convert to the dtype if needed
        if model == 'gbm':
            time_str, hist_wdw = extract_file_name_gbm(uid)
            print(time_str, hist_wdw)
        else:
            date_time, hist_wdw, max_sigma = extract_file_name_heston(uid)
            print(date_time, hist_wdw, max_sigma)

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
        pd.set_option('display.max_rows', None)
        logger_yq.info(f"The payouts compare df is: {payouts_compare}")
        pd.reset_option('display.max_rows')

        # For Heston need to deal with empty PPD, calculate 1 more RMSE_clean
        if model == 'heston':
            compare_clean = payouts_compare.copy(deep=True)
            compare_clean = compare_clean[(compare_clean.index < pd.Timestamp('2023-09-15')) | 
                                          (compare_clean.index > pd.Timestamp('2023-10-03'))]
            compare_clean = compare_clean[compare_clean['ppd_payouts'] > 0]
            RMSE_clean = np.sqrt(np.mean((compare_clean['ppd_payouts'] - compare_clean['actual_price']) ** 2))

        RMSE = np.sqrt(np.mean((payouts_compare['ppd_payouts'] - payouts_compare['actual_price']) ** 2))
        MAE = np.mean(np.abs(payouts_compare['ppd_payouts'] - payouts_compare['actual_price']))
        print(f"MAE: {MAE}")
        
        if model == 'heston':
            RMSE_dict[f"{model}_{uid}_{n_sim}_{n_ppd}_unadj"] = RMSE
            RMSE_dict[f"{model}_{uid}_{n_sim}_{n_ppd}_adj"] = RMSE_clean

            logger_yq.info(f"RMSE and RMSE_clean for {model}_{uid}_{n_sim}_{n_ppd} is:\n{RMSE} {RMSE_clean}")
            print(RMSE, RMSE_clean)
            title_str = f"model, hist_wdw, max_sigma"

            compare_list = [payouts_compare, compare_clean]
            adjusted = ['unadj', 'adj']
            for i, rmse_val in enumerate([RMSE, RMSE_clean]):
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.tight_layout()
                title_str = f"Model: {model}, hist_wdw: {hist_wdw}, adj: {adjusted[i]}"
                subtitle_str = f"n_sim: {n_sim}, n_ppd: {n_ppd}, max_sigma: {max_sigma}, RMSE: {rmse_val:.2f}"
                plt.title(f"{title_str}\n{subtitle_str}")  # Adjust font size as needed
                
                ax.plot(compare_list[i].index, compare_list[i]['ppd_payouts'],
                        alpha = 1, label='ppd_payouts') # Must use this to see title
                ax.plot(compare_list[i].index, compare_list[i]['actual_price'],
                        alpha = 1, label='actual_price')
                plt.legend(loc='upper right')

                stor_dir = yq_path.get_plots_path(Path(__file__).parent).joinpath('eval', model)
                file_path = stor_dir.joinpath(f'{model}_{uid}_{n_sim}_{n_ppd}_{adjusted[i]}.png')
                stor_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
        else:
            RMSE_dict[f"{model}_{uid}_{hist_wdw}_{n_sim}_{n_ppd}"] = RMSE
            # logger_yq.info(f"RMSE for combination is:\n{RMSE} {RMSE_clean}")

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

    logger_yq.info(f"RMSE dict is {RMSE_dict}")
    print(RMSE_dict)


def analyse_RMSE_asset():
    """
    Failed attempt. Only works for small value of n_sim. Large amount will cause
    the mean path to be a horizontal straight line. But actually even the small 
    value of n_sim also cannot capture the trend. If the model is able to capture,
    most of the paths will lie around the mean (due to normal distribution) which 
    is still a good representation of the average path.
    """
        # Function parameters
    model = 'gbm'
    gbm_files = ["20231114_092701_252"] # Top performers: GBM: 20231114_092701_252 heston 20231115_032413_252_1.5 # TODO: heston need adjust
    # Test GBM: 20231114_024525_7
    # RMSE dict for each ppd
    RMSE_dict = {}
    RMSE_list = []
    for uid in gbm_files:

        print(uid)
        # Getting back the strings, rmb to convert to the dtype if needed
        if model == 'gbm':
            time_str, hist_wdw = extract_file_name_gbm(uid)
            print(time_str, hist_wdw)
        else:
            date_time, hist_wdw, max_sigma = extract_file_name_heston(uid)
            print(date_time, hist_wdw, max_sigma)

        paths_arr, dates = sm.read_sim_data(model, uid, cs.INITIAL_PROD_PRICING_DATE, cs.FINAL_PROD_PRICING_DATE)
        n_sim = len(paths_arr[0])
        n_ppd = len(paths_arr) # Some missing lists in n_ppd
        print(n_sim, n_ppd)

        # For each ppd/simulation date, we have many paths. Take average, each asset with its asset price
        # The dates for the path need to be sliced (first day of sim to last day of avai data by sc func)
        for ppd in range(1): # TODO: Do for 1 path first
            actual_St = po.get_historical_assets_all()
            sim_start_date = paths_arr[ppd][0].first_valid_index()
            actual_St = actual_St[(actual_St.index >= sim_start_date)]
            actual_St.columns = [f'actual_{name}' for name in actual_St.columns]
            avai_end_date = actual_St.last_valid_index()
            print(avai_end_date)
            #display(actual_St.head())
            # Concat all the UA prediction across all sims
            # Then take all the sims average groupby the date of sim
            paths = pd.DataFrame() 

            plt.subplots(figsize=(10, 6))
            plt.tight_layout()
            plt.tight_layout() # TODO:
            title_str = f"Simulated Paths Against Actual Share Price"
            subtitle_str = f"Model: {model}, hist_wdw: {hist_wdw}, n_sim: {n_sim}, n_ppd: {n_ppd}"
            plt.title(f"{title_str}\n{subtitle_str}")  # Adjust font size as needed
                
            # Concat and TODO: plot graph
            for sim in range(n_sim):
                path = paths_arr[ppd][sim]
                path = path[path.index <= avai_end_date]
                #display(f"Path:\n{path}")
                paths = pd.concat([paths, path], axis=0)

                for asset in cs.ASSET_NAMES:
                    plt.plot(path.index, path[asset], alpha=0.5, label=f'sim_{asset}')  

            for asset in cs.ASSET_NAMES:
                plt.plot(actual_St.index, actual_St, alpha=1, label=f'actual_{asset}')
            # path.plot()
            stor_dir = yq_path.get_plots_path(Path(__file__).parent).joinpath('eval', model)
            file_path = stor_dir.joinpath(f'{model}_{uid}_{n_sim}_{n_ppd}_asset.png')
            stor_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(file_path, bbox_inches='tight')

            plt.close()

            #display(len(paths) == len(actual_St) * n_sim)
            #display(len(actual_St))
            #display(len(paths))

            # Get average
            paths_mean = paths.groupby(paths.index).mean()

            # Concat axis=1 with actual price
            if (len(paths_mean) == len(actual_St)):
                St_compare = pd.concat([paths_mean, actual_St], axis=1)
                # display(St_compare.head())
            else:
                print("Error with length")

            # Get RMSE
            for asset in cs.ASSET_NAMES:
                RMSE  = np.sqrt(np.mean((St_compare[asset] - 
                                        St_compare[f'actual_{asset}']) ** 2))
                RMSE_dict[f'{asset}_RMSE'] = RMSE
                RMSE_list.append(RMSE.round(2))
                
                
            # display(RMSE_dict)
            plt.subplots(figsize=(10, 6))
            plt.tight_layout()
            plt.tight_layout() # TODO:
            title_str = f"Simulated Paths Against Actual Share Price"
            subtitle_str = f"Model: {model}, hist_wdw: {hist_wdw}, n_sim: {n_sim}, n_ppd: {n_ppd}, RMSE: {RMSE_list}"
            plt.title(f"{title_str}\n{subtitle_str}")  # Adjust font size as needed

            for asset in cs.ASSET_NAMES:
                plt.plot(paths_mean.index, paths_mean[asset], alpha=0.5, label=f'sim_{asset}')  

            for asset in cs.ASSET_NAMES:
                plt.plot(actual_St.index, actual_St, alpha=1, label=f'actual_{asset}')
            # path.plot()
            plt.legend(loc='upper right')
            stor_dir = yq_path.get_plots_path(Path(__file__).parent).joinpath('eval', model)
            file_path = stor_dir.joinpath(f'{model}_{uid}_{n_sim}_{n_ppd}_mean_asset.png')
            stor_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(file_path, bbox_inches='tight')

            plt.close()
            plt.close()

            





