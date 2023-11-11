from pathlib import Path
import pandas as pd
from yq.utils import calendar

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
    curr_dir = Path("/Users/tangyiqwan/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/ntu/Acads/4_Y4S1/MH4518/group-project/code/simulation-in-finance/code/yq/playground_1.ipynb").parent.parent.parent
    # curr_dir = Path(__file__).parent.parent.parent

    print(curr_dir)

    file_path = curr_dir.joinpath('data', 'options', file_name)
    print(file_path)

    options_data = pd.read_csv(file_path)

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
    options_data['rate'] = [1.75/100 for _ in range(len(options_data))]

    return options_data

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

if __name__ == "__main__":
    pass
    # create_date_folders(pd.Timestamp('2023-08-09'))
    # lonn_call = read_options_data("lonn_call.csv")
    # print(lonn_call)

    # sika_call = read_options_data("sika_call.csv")
    # print(sika_call)