import datetime
import logging
import pandas as pd
import typing
from pathlib import Path
from yq.utils import calendar

logger_yq = logging.getLogger('yq')

def read_sim_data(model_name: str, 
                  uid: str, 
                  prod_est_start_date: pd.Timestamp, 
                  prod_est_end_date: pd.Timestamp) -> typing.List:
    """
    Returns the a 2D array paths_arr[PPD][sim] and a list of PPD available
    For Heston might not be 67 because of Cholesky issues
    """
    logger_yq.info('Reading sim data')
    cur_dir = Path(__file__).parent
    trading_calendar = calendar.SIXTradingCalendar()
    bus_date_range = trading_calendar.create_six_trading_dates(prod_est_start_date, prod_est_end_date)
    product_est_date_sim_data_df_list, dates = [], []
    for product_est_date in bus_date_range.index:
        storage_dir = cur_dir.joinpath('..', '..', '..', 'sim_data', model_name, uid, product_est_date.strftime('%Y-%m-%d'))

        # Check if storage_dir exists
        if not storage_dir.exists():
            logger_yq.error(f"Storage directory not found: {storage_dir}")
            continue  # Skip to the next iteration
        
        file_count = len([file for file in storage_dir.glob('*') if file.is_file()])
        #logger_yq.info(f'The number of files is {file_count}')
        sim_data_df = []
        for sim in range(file_count):
            file_path = storage_dir.joinpath(str(sim) + '.csv')
            # logger_yq.info(f'The file path to read sim data is {file_path}')
            try:
                new_df = pd.read_csv(file_path)
                new_df['Date'] = pd.to_datetime(new_df['Date'])
                new_df = new_df.set_index('Date')
                sim_data_df.append(new_df)
            except FileNotFoundError:
                logger_yq.error(f'File not found: {file_path}')
            except Exception as e:
                raise

        dates.append(product_est_date)
        product_est_date_sim_data_df_list.append(sim_data_df)
        # print(f"sim_data_df for {product_est_date}:\n {product_est_date_sim_data_df_list}\n")
        logger_yq.info(f"Total sims/length of sim_data_df for {product_est_date}: {len(sim_data_df)}")

    logger_yq.info(f"Total days is: {len(product_est_date_sim_data_df_list)}")
    return product_est_date_sim_data_df_list, dates

def store_sim_data(uid: str,
                   model_name: str,
                   sim_data: pd.DataFrame,
                   product_est_date: pd.Timestamp,
                   sim: int) -> None:
    cur_dir = Path(__file__).parent
    storage_dir = cur_dir.joinpath('..', '..', '..', 'sim_data', model_name, uid, product_est_date.strftime('%Y-%m-%d'))
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir.joinpath(str(sim) + '.csv')
    sim_data.to_csv(file_path)

if __name__ == "__main__":
    product_est_date_sim_data_df_list = read_sim_data(
        model_name='gbm',
        uid='20231110_222722_149946', 
        prod_est_start_date=pd.Timestamp('2023-08-09'), 
        prod_est_end_date=pd.Timestamp('2023-08-10'))
    print(product_est_date_sim_data_df_list)