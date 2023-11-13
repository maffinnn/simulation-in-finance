import logging
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from yq.utils import option
from yq.utils import path as yq_path
from sc import payoff as po
from sc import constants as cs

logger_yq = logging.getLogger('yq')

def analyse_V_t():
    # Sample data from the user input
    curr_dir = Path(__file__)
    file_path = yq_path.get_root_dir(cur_dir=curr_dir).joinpath('code', 'yq', 'inter-data', '20230809_max_sigma_0.5_v_t.txt')

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
    plt.title('V_t Values Over Time\nDate: 2023-08-09, max_sigma = 0.5, kappa = [3.70, 8.30]')
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

