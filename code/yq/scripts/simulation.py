import datetime
import pandas as pd
import typing
from pathlib import Path
from yq.utils import calendar


def read_sim_data(model_name: str, 
                  start_time_str: str, 
                  prod_est_start_date: pd.Timestamp, 
                  prod_est_end_date: pd.Timestamp) -> typing.List:

    
    cur_dir = Path(__file__).parent
    trading_calendar = calendar.SIXTradingCalendar()
    bus_date_range = trading_calendar.create_six_trading_dates(prod_est_start_date, prod_est_end_date)
    product_est_date_sim_data_df_list = []
    for product_est_date in bus_date_range.index:
        storage_dir = cur_dir.joinpath('..', '..', '..', 'sim_data', model_name, start_time_str, product_est_date.strftime('%Y-%m-%d'))
        file_count = len([file for file in storage_dir.glob('*') if file.is_file()])
        sim_data_df = []
        for sim in range(file_count):
            file_path = storage_dir.joinpath(str(sim) + '.csv')
            sim_data_df.append(pd.read_csv(file_path))

        product_est_date_sim_data_df_list.append(sim_data_df)
        # print(f"sim_data_df for {product_est_date}:\n {product_est_date_sim_data_df_list}\n")
        print(f"Total sims/length of sim_data_df for {product_est_date}: {len(sim_data_df)}")
    print(f"Total days is: {len(product_est_date_sim_data_df_list)}\n")
    return product_est_date_sim_data_df_list

def store_sim_data(start_time_acc: datetime,
                   model_name: str,
                   sim_data: pd.DataFrame,
                   product_est_date: pd.Timestamp,
                   sim: int) -> None:
    start_time_str = start_time_acc.strftime('%Y%m%d_%H%M%S_%f')
    model_name = 'gbm'
    cur_dir = Path(__file__).parent
    storage_dir = cur_dir.joinpath('..', '..', '..', 'sim_data', model_name, start_time_str, product_est_date.strftime('%Y-%m-%d'))
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir.joinpath(str(sim) + '.csv')
    sim_data.to_csv(file_path)

if __name__ == "__main__":
    product_est_date_sim_data_df_list = read_sim_data(
        model_name='gbm',
        start_time_str='20231110_222722_149946', 
        prod_est_start_date=pd.Timestamp('2023-08-09'), 
        prod_est_end_date=pd.Timestamp('2023-08-10'))
    print(product_est_date_sim_data_df_list)