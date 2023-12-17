from yq.utils import path as yq_path
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger_yq = logging.getLogger('yq')

def write_hparams(prod_date: pd.Timestamp, max_sigma: float, hparams_list: np.array):
    """
    This function saves a given numpy array of hyperparameters to a CSV file, named after the production date. 
    It creates a directory if it doesn't exist, based on the provided max_sigma value.

    Parameters:
    prod_date (pd.Timestamp): The product pricing date used in the filename.
    max_sigma (float): The max sigma used in Heston's calibration.
    hparams_list (np.array): Numpy array of hyperparameters to be saved.

    Raises:
    Exception: If there is any issue in saving the hyperparameters to the CSV file.
    """
    try:
        # Get the directory path for storing hyperparameters
        stor_dir = yq_path.get_hparams(cur_dir=Path(__file__)).joinpath(str(max_sigma))
        stor_dir.mkdir(parents=True, exist_ok=True)
        file_path = stor_dir.joinpath(f"{prod_date.strftime('%Y%m%d')}.csv")
        
        # Convert numpy array to DataFrame for easier CSV writing
        pd.DataFrame(hparams_list).to_csv(file_path, index=False)
        logger_yq.info(f"Parameters saved successfully to {file_path}")
    except Exception as e:
        logger_yq.error(f"Failed to save hyperparameters: {e}")
        raise


def read_hparams(prod_date: pd.Timestamp, max_sigma: float) -> np.array:
    """
    This function reads a CSV file into a numpy array. The CSV file is expected to be named after the 
    production date and located within a directory named after the max_sigma value.

    Parameters:
    prod_date (pd.Timestamp): The product pricing date used in the filename.
    max_sigma (float): The max sigma used in Heston's calibration.

    Returns:
    np.array: Numpy array of hyperparameters read from the CSV file.

    Raises:
    FileNotFoundError: If the CSV file does not exist.
    Exception: If there is an error during file reading.
    """
    try:
        # Construct the file path for the CSV containing hyperparameters
        file_path = yq_path.get_hparams(cur_dir=Path(__file__)).joinpath(str(max_sigma), 
                                                                         f'{prod_date.strftime("%Y%m%d")}.csv')
        # Read CSV file into DataFrame and then convert to numpy array
        hparams_list = pd.read_csv(file_path).values
        logger_yq.info(f"Parameters loaded successfully from {file_path}")
    except FileNotFoundError:
        logger_yq.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger_yq.error(f"An error occurred while reading the file: {e}")
        raise

    return hparams_list

def test_io():
    """
    This function tests the write_hparams() and read_hparams() functions.
    """
    sample_params = np.array([
        [6.28462244, 0.08097303, 0.06545346, -0.29282084, 1.97786912], 
        [8.06772349, 0.09460271, 0.09482759, -0.41860207, 3.81423388]
    ])
    dates = [pd.Timestamp('2023-08-09')]
    for date in dates:
        write_hparams(date, sample_params)
        print(read_hparams(date))