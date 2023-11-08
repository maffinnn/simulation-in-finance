import typing
import pandas as pd
import constants as cs
import pandas_market_calendars as mcal

# Takes the start and end dates to create a trading calendar for SIX
# Returns a df with all the dates of trading days as index
def create_six_trading_dates(start_date, end_date):
    six_calendar = mcal.get_calendar('SIX')
    print(f"Holidays in the calendar up to 2200: {six_calendar.holidays().holidays[-10:]}")
    six_trading_days = six_calendar.valid_days(start_date, end_date).tz_localize(None)

    six_trading_days_df = pd.DataFrame(index = six_trading_days)
    
    # Name the index "Dates"
    six_trading_days_df.index.name = 'Dates'
    return six_trading_days_df

def add_trading_day(trading_date: pd.Timestamp, num_trading_day: int) -> pd.Timestamp:
    """
    Adds a specified number of trading days to a given date within the range of 2020 to 2025.

    Parameters:
    trading_date (pd.Timestamp): The trading date from which to add trading days.
    num_trading_day (int): The number of trading days to add, can be negative.

    Returns:
    pd.Timestamp: The resulting date after adding the trading days.

    Raises:
    ValueError: If the `trading_date` is not a valid trading date or out of the date range.
    IndexError: If the resulting trading date is out of the valid date range.
    """

    # Assuming create_six_trading_dates returns a pd.DatetimeIndex with trading dates
    trading_df = create_six_trading_dates('2020-01-01', '2025-12-31')
    print(trading_df)
    try:
        position = trading_df.index.get_loc(trading_date)
        print(position)
    except:
        raise ValueError(f"{trading_date} is not a trading date or is out of the given date range.")

    new_position = position + num_trading_day

    try:
        return trading_df.index[new_position] # Returns the index of the position (cannot use iloc)
    except:
        raise IndexError("The resulting trading date is out of bounds.")

# Example usage:
# new_trading_date = add_trading_day(pd.Timestamp('2023-01-10'), 5)
# print(new_trading_date)

# Input: a dataframe ith dates as index
def remove_SIX_holidays(data: pd.DataFrame) -> pd.DataFrame:
    # Ensure that the dates you are trying to drop exist in the index
    dates_to_drop = [date for date in cs.SIX_HOLIDAY_DATES if date in data.index]
    print(f"The dates to drop are: {dates_to_drop}")

    # Drop the dates
    dropped_data = data.drop(dates_to_drop)
    # print(dropped_data)
    return dropped_data

