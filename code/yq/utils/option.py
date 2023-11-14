import logging
from pathlib import Path
import pandas as pd
from yq.utils import calendar, path as yq_path
# from yq import logs
from sc import constants as cs

logger_yq = logging.getLogger('yq')

def read_clean_options_data(options_dir: str, curr_date: pd.Timestamp, file_name: str) -> pd.DataFrame:
    curr_dir = Path(__file__).parent
    root_dir = yq_path.get_root_dir(curr_dir)
    dir_name = curr_date.strftime('%Y%m%d')
    file_path = root_dir.joinpath('data', options_dir, dir_name, file_name)

    options_data = pd.read_csv(file_path, index_col=None)
    logger_yq.info("File path is %s", file_path)
    logger_yq.info(f"Options data read is\n {options_data.head()}")
    return options_data

def clean_options_data(options_dir: str):
    # Read CSV
    trading_cal = calendar.SIXTradingCalendar()
    dates_df = trading_cal.create_six_trading_dates(cs.INITIAL_PROD_PRICING_DATE, cs.FINAL_PROD_PRICING_DATE)
    cur_dir = Path(__file__).parent
    root_dir = yq_path.get_root_dir(cur_dir)
    for date in dates_df.index:
        dir_name = date.strftime('%Y%m%d')
        tar_dir = root_dir.joinpath('data', options_dir, dir_name)
        store_dir = root_dir.joinpath('data', 'options-cleaned', dir_name)
        store_dir.mkdir(parents=True, exist_ok=True)
        logger_yq.info("Storage directory is %s", store_dir)
        logger_yq.info("Folder name is: %s", date.strftime('%Y%m%d'))

        if tar_dir.exists() and tar_dir.is_dir():
            files = list(tar_dir.glob('*.csv'))
            if len(files) != 2: 
                logger_yq.info("Warning: %s has %s CSV files.", date, len(files))

            # Number of CSV files is exactly 2
            lonza_cnt, sika_cnt = 0, 0
            for file in files:
                file_name = file.name
                logger_yq.info(f"The file name is {file_name}")
                with open(file, 'r') as f:
                    for line in f:
                        if file_name == 'lonn_call.csv' and 'LONE' in line:
                            lonza_cnt += 1 # Must be 1 if things are right
                            break
                        elif file_name == 'sika_call.csv' and 'SIK' in line: 
                            sika_cnt += 1
                            break
                options_data = pd.read_csv(file, index_col=None)
                cle_opt_data = clean_options_df(options_data=options_data, curr_date=date)
                logger_yq.info(f"The cle_opt_data df is:\n{cle_opt_data.head()}")
                new_path = store_dir.joinpath(str(file_name))
                logger_yq.info(f"The new path is {new_path}")
                cle_opt_data.to_csv(new_path, index=False)

            if (lonza_cnt != 1) or (sika_cnt != 1):
                logger_yq.warning(f"Warning: Data might not belong to the asset in {date}'s files.")
                continue
        else:
            logger_yq.warning(f"Warning: Path not exist or not a folder")


def clean_options_df(options_data: pd.DataFrame, curr_date: pd.Timestamp):
    # Drop the description rows
    options_data.dropna(subset=['ExDt'], inplace=True)
    # print(options_data.isna().sum())
    
    # Calculate time to maturity
    options_data['ExDt'] = pd.to_datetime(options_data['ExDt'], format='%m/%d/%y')
    options_data['maturity'] = (options_data['ExDt'] - pd.to_datetime(curr_date)).dt.days / 356.25

    # Remove rows with zeros
    options_data = options_data[(options_data['Mid'] > 0) & (options_data['IVM'] > 0)].reset_index(drop=True)

    options_data = options_data[['maturity', 'Strike', 'Mid', 'IVM']]
    options_data.columns = ['maturity', 'strike', 'price', 'IV']

    options_data = options_data[options_data['maturity'] > 0.1]
    
    # Calculate the risk free rate for each maturity
    options_data['rate'] = [cs.INTEREST_RATE for _ in range(len(options_data))] 

    if (options_data.isna().sum().sum() > 0 or (options_data == 0).sum().sum() > 0):
        logger_yq.warning(f"Options data contain 0 or NaN values")
    elif (len(options_data) == 0):
        logger_yq.error(f"Options data became empty")
    return options_data

def format_file_names(options_dir: str):
    # Change the options-test to other file names for actual op
    trading_cal = calendar.SIXTradingCalendar()
    dates_df = trading_cal.create_six_trading_dates(cs.INITIAL_PROD_PRICING_DATE, cs.FINAL_PROD_PRICING_DATE)
    
    cur_dir = Path(__file__).parent
    root_dir = yq_path.get_root_dir(cur_dir)
    for date in dates_df.index:
        dir_name = date.strftime('%Y%m%d')
        tar_dir = root_dir.joinpath('data', options_dir, dir_name)
        logger_yq.info("Target options directory is %s", tar_dir)
        logger_yq.info("Folder name is: %s", date.strftime('%Y%m%d'))

        if tar_dir.exists() and tar_dir.is_dir():
            files = list(tar_dir.glob('*'))
            if len(files) != 2: 
                logger_yq.info("Warning: %s has %s files.", date, len(files))

            # Number of files is exactly 2
            lonza_cnt, sika_cnt = 0, 0
            for file in files:
                file_name = file.name
                logger_yq.info(f"The file name is {file_name}")
                # Skip the renaming if the two file names are already inside
                if file_name == 'lonn_call.csv':
                    lonza_cnt += 1
                elif file_name == 'sika_call.csv':
                    sika_cnt += 1
            if (lonza_cnt == 1) and (sika_cnt == 1):
                logger_yq.info("lonn_call and sika_call ady exist, end iteration.")
                continue

            rename_count = 0
            for file in files:
                file_name = file.name
                # Change the file names if the two target names don't exist
                if file_name == 'grid1.xlsx':
                    new_fname = tar_dir.joinpath('lonn_call.csv')
                    df = pd.read_excel(file, engine='openpyxl') # Need to use pandas to convert
                    df.to_csv(new_fname, index=False)
                    file.unlink()
                    logger_yq.info("Renamed to lonn_call.csv successfully")
                    rename_count += 1
                elif file_name.startswith('grid1_'):
                    new_fname = tar_dir.joinpath('sika_call.csv')
                    df = pd.read_excel(file, engine='openpyxl') # Need to use pandas to convert
                    df.to_csv(new_fname, index=False)
                    file.unlink()
                    logger_yq.info("Renamed to sika_call.csv successfully")
                    rename_count += 1
            if rename_count != 2:
                logger_yq.info("Warning: Did not perform 2 renames")

        else:
            logger_yq.info("Path not exist or not a folder")

def check_data():
    pass

def create_csv_files(prod_est_date: pd.Timestamp) -> None:
    """
    Create a directory structure for a given production estimated date and
    creates two CSV files, 'lonn_call.csv' and 'sika_call.csv', in this directory.

    Parameters:
    prod_est_date (pd.Timestamp): The production estimated date.

    The function creates a directory path in the format 'data/options/YYYYMMDD' 
    relative to the script's location and creates two CSV files in it.
    """
    try:
        cur_dir = Path(__file__).parent
        target_dir = cur_dir.joinpath('..', '..', '..', 'data', 'options', prod_est_date.strftime('%Y%m%d'))
        target_dir.mkdir(parents=True, exist_ok=True)

        # Creating CSV file paths using joinpath
        lonn_call_path = target_dir.joinpath('lonn_call.csv')
        sika_call_path = target_dir.joinpath('sika_call.csv')

        # Create two empty CSV files
        lonn_call_path.touch()
        sika_call_path.touch()

        # Example: Writing headers to CSV files
        # with open(lonn_call_path, 'w') as f:
        #     f.write('Header1,Header2,Header3\n')
        # with open(sika_call_path, 'w') as f:
        #     f.write('Header1,Header2,Header3\n')

    except Exception as e:
        print(f"An error occurred: {e}")

def read_options_data(file_name: str):
    """
    Read the option data from a CSV file and clean it.

    Parameters:
    - file_name (str): The file name of the CSV file located in the 'data/options' folder.

    Return:
    - options_data (pd.DataFrame): A DataFrame with only three columns: 'maturity',
      'strike', and 'price'.

    Notes:
    - The function expects CSV files to have specific columns, including 'ExDt' and 'Mid'.
    - Rows with missing 'ExDt' values and rows where 'Mid' is zero are removed.
    - 'maturity' is calculated as the number of days from 'ExDt' to a specific date 
      (here '2023-11-07'), divided by 356.25. The significance of 356.25 should be 
      understood in the context of the analysis.

    Warning:
    - In Jupyter notebooks, the __file__ attribute is not available. The current 
      implementation uses a hard-coded path for the Jupyter environment. For script 
      execution, use `Path(__file__).parent.parent.parent` to set the correct path.
    """
    # curr_dir = Path("/Users/tangyiqwan/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/ntu/Acads/4_Y4S1/MH4518/group-project/code/simulation-in-finance/code/yq/playground_1.ipynb").parent.parent.parent
    curr_dir = Path(__file__).parent.parent.parent.parent

    print(curr_dir)

    file_path = curr_dir.joinpath('data', 'options', file_name)
    print(file_path)

    options_data = pd.read_csv(file_path, index_col=None)

    # Drop the description rows
    # display(options_data)
    options_data.dropna(subset=['ExDt'], inplace=True)
    # display(options_data.isna().sum())
    
    # Calculate time to maturity
    options_data['ExDt'] = pd.to_datetime(options_data['ExDt'], format='%m/%d/%y')
    options_data['maturity'] = (options_data['ExDt'] - pd.to_datetime('2023-11-07')).dt.days / 356.25

    # Remove rows with zeros
    options_data = options_data[options_data['Mid'] > 0].reset_index(drop=True)

    options_data = options_data[['maturity', 'Strike', 'Mid']]
    options_data.columns = ['maturity', 'strike', 'price']

    # display(options_data)
    
    # Calculate the risk free rate for each maturity
    options_data['rate'] = [cs.INTEREST_RATE for _ in range(len(options_data))]

    return options_data
if __name__ == "__main__":
    pass
    # create_date_folders(pd.Timestamp('2023-08-09'))
    # lonn_call = read_options_data("lonn_call.csv")
    # print(lonn_call)

    # sika_call = read_options_data("sika_call.csv")
    # print(sika_call)