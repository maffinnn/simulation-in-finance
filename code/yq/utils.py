import typing
import pandas as pd
import constants as cs

# Input: a dataframe ith dates as index
def remove_SIX_holidays(data: pd.DataFrame) -> pd.DataFrame:
    # Ensure that the dates you are trying to drop exist in the index
    dates_to_drop = [date for date in cs.SIX_HOLIDAY_DATES if date in data.index]
    print(f"The dates to drop are: {dates_to_drop}")

    # Drop the dates
    dropped_data = data.drop(dates_to_drop)
    # print(dropped_data)
    return dropped_data