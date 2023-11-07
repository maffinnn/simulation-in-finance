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

# Input: trading date (assumed) and number of trading days to add (can be positive or negative values)
# Limitation is to range of dates available is between 2020 and 2025
def add_trading_day(trading_date, num_trading_day):
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

# Input: a dataframe ith dates as index
def remove_SIX_holidays(data: pd.DataFrame) -> pd.DataFrame:
    # Ensure that the dates you are trying to drop exist in the index
    dates_to_drop = [date for date in cs.SIX_HOLIDAY_DATES if date in data.index]
    print(f"The dates to drop are: {dates_to_drop}")

    # Drop the dates
    dropped_data = data.drop(dates_to_drop)
    # print(dropped_data)
    return dropped_data

