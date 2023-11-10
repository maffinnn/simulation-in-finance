import typing
import pandas as pd
# from ...sc import constants as cs
import pandas_market_calendars as mcal

class SIXTradingCalendar:
    def __init__(self) -> None:
        self.six_trading_calendar = self.create_six_trading_dates('2020-01-01', '2025-12-31')
        
    def calculate_business_days(self, start_date, end_date):
        """
        Calculate the number of SIX Swiss Exchange business days between two dates, inclusive.
        The start and end date can be non-business days.

        Parameters:
            start_date (pd.Timestamp): The start date for the calculation.
            end_date (pd.Timestamp): The end date for the calculation.

        Returns:
            int: The number of business days between start_date and end_date, inclusive.
        """
        six_calendar = mcal.get_calendar('SIX')
        return len(six_calendar.valid_days(start_date, end_date).tz_localize(None))

    def create_six_trading_dates(self, start_date, end_date):
        """
        Create a DataFrame of trading dates for the SIX Swiss Exchange within a given date range.

        Parameters:
            start_date (str): The start date (inclusive) to generate trading dates from in 'YYYY-MM-DD' format.
            end_date (str): The end date (inclusive) to generate trading dates to in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame where each row represents a trading day within the date range.
                          The DataFrame's index is named 'Dates' and contains the trading dates.
        """
        six_calendar = mcal.get_calendar('SIX')
        print(f"Holidays in the calendar up to 2200: {six_calendar.holidays().holidays[-10:]}")
        six_trading_days = six_calendar.valid_days(start_date, end_date).tz_localize(None)

        six_trading_days_df = pd.DataFrame(index = six_trading_days)
        
        # Name the index "Dates"
        six_trading_days_df.index.name = 'Date'
        return six_trading_days_df

    def add_trading_day(self, trading_date: pd.Timestamp, num_trading_day: int) -> pd.Timestamp:
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
        try:
            position = self.six_trading_calendar.index.get_loc(trading_date)
            print(position)
        except:
            raise ValueError(f"{trading_date} is not a trading date or is out of the given date range.")

        new_position = position + num_trading_day

        try:
            return self.six_trading_calendar.index[new_position] # Returns the index of the position (cannot use iloc)
        except:
            raise IndexError("The resulting trading date is out of bounds.")

    # # Input: a dataframe ith dates as index
    # def remove_SIX_holidays(data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Removes rows from the DataFrame that correspond to SIX Swiss Exchange holidays.

    #     This method assumes that the DataFrame's index is a DatetimeIndex with dates that
    #     potentially include holidays. It filters out any dates that are present in the
    #     predefined list of SIX holidays.

    #     Parameters:
    #         data (pd.DataFrame): A DataFrame with dates as its index, which represents
    #                              trading days including holidays.

    #     Returns:
    #         pd.DataFrame: A DataFrame with the same structure as the input but with
    #                       holidays removed from the index.

    #     Note:
    #         The SIX holiday dates are obtained from a predefined list `cs.SIX_HOLIDAY_DATES`.
    #     """
    #     # Ensure that the dates you are trying to drop exist in the index
    #     dates_to_drop = [date for date in cs.SIX_HOLIDAY_DATES if date in data.index]
    #     print(f"The dates to drop are: {dates_to_drop}")

    #     # Drop the dates
    #     dropped_data = data.drop(dates_to_drop)
    #     # print(dropped_data)
    #     return dropped_data

if __name__ == "__main__":
    # Create an instance of the SIXTradingCalendar class
    trading_calendar = SIXTradingCalendar()

    # Perform various operations using the class instance
    # For example, calculate the number of business days between two dates
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-01-10')
    business_days = trading_calendar.calculate_business_days(start_date, end_date)
    print(f"Number of business days: {business_days}")
    