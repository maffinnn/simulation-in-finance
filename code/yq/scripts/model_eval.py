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

@timeit
# hist_windows = [63]
#     n_sims = [3]
#     models = ['gbm','heston']
#     max_sigmas = [0.5, 1.5, 10]
def analyse_rmse(hist_windows: list, n_sims: list, models: list, max_sigmas: list):
    # TODO: Take the values from yq_script
    model = '' #TODO:
    dir_list = ['20231113_185603_63_0.5']


    # For different methodologies, we want to get the RMSE for the ppd_payous against actual price
    # Change to itertools
    RMSE_list = []
    for uid in dir_list:
        paths_arr = sm.read_sim_data(model, uid, cs.INITIAL_PROD_PRICING_DATE, cs.FINAL_PROD_PRICING_DATE)
        n_sim = len(paths_arr[0])
        n_ppd = len(paths_arr)
    

        actual_price = po.get_product_price(cs.FINAL_PROD_PRICING_DATE).rename(columns={'Price': 'actual_price'})
        # Actual price for the product price period
        actual_price = actual_price[actual_price.index >= cs.INITIAL_PROD_PRICING_DATE]

        # Average payouts of all the sim paths on each ppd
        ppd_payouts = []
        for ppd in range(n_ppd):
            # Payouts for all the paths on one day of the price period (multiple paths)
            paths_payout = po.pricing_multiple(paths_arr[ppd])
            ppd_payouts.append(np.mean(paths_payout))
        # Payouts for the entire pricing period
        
        if (len(ppd_payouts) != len(actual_price)):
            logger_yq.error(f"The length of payouts and actual price dfs is diff: {len(ppd_payouts)}, {len(actual_price)}")
        
        ppd_payouts_df = pd.DataFrame({'ppd_payouts': ppd_payouts})
        ppd_payouts_df.index = actual_price.index
        payouts_compare = pd.concat([ppd_payouts_df, actual_price], axis=1)
        logger_yq.info(f"The payouts compare df is:\n{payouts_compare.head()}")

        # # TODO: Plot payouts_compare
        # fig, ax = plt.subplots(figsize=(10,6)) 
        # if self.model_name == 'heston':
        #     title_str = f"Model: {model}, hist_wdw: INSERTSTH{model}, max_sigma: {max_sigma}"
        # else:
        #     # TODO: Do sth for gbm
        #     pass
        # subtitle_str = f"xxx"
        # plt.title(f"{title_str}\n{subtitle_str}")

        # payouts_compare.plot()
        # plt.legend(loc='upper right')

        # stor_dir = yq_path.get_plots_path(Path(__file__).parent).joinpath(model, 'eval')
        # stor_dir.mkdir(parents=True, exist_ok=True)
        # file_path = stor_dir.joinpath(f'#############.png') # TODO: Rename
        # plt.savefig(file_path, bbox_inches='tight')
        # plt.close()

        # RMSE = np.sqrt(np.mean((payouts_compare['ppd_payouts'] - payouts_compare['actual_price']) ** 2))
        # RMSE_list.append(RMSE)
        # logger_yq.info(f"RMSE for combination is:\n{RMSE}") # TODO: Add combination

    # TODO: Write the list into a csv file or sth



