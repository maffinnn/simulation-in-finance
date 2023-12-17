import typing
import pandas as pd
import pandas_market_calendars as mcal

# Custom class for handling trading calendar for SIX Swiss Exchange
class SIXTradingCalendar:
    def __init__(self) -> None:
        # Initialize the class by creating a DataFrame with SIX trading dates for a given range
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
        # Retrieve the SIX trading calendar
        six_calendar = mcal.get_calendar('SIX')
        # Get the number of valid trading days within the given date range
        return len(six_calendar.valid_days(start_date, end_date).tz_localize(None))

    def create_six_trading_dates(self, start_date, end_date):
        """
        Create a DataFrame of trading dates for the SIX Swiss Exchange within a given date range.
        
        Parameters:
            start_date (str): The start date (inclusive) to generate trading dates from in 'YYYY-MM-DD' format.
            end_date (str): The end date (inclusive) to generate trading dates to in 'YYYY-MM-DD' format.
        
        Returns:
            pd.DataFrame: A DataFrame where each row represents a trading day within the date range.
                          The DataFrame's index is named 'Date' and contains the trading dates.
        """
        # Retrieve the SIX trading calendar
        six_calendar = mcal.get_calendar('SIX')
        # print(f"Holidays in the calendar up to 2200: {six_calendar.holidays().holidays[-10:]}")
        # Generate a list of valid trading days within the specified date range
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
        # Check if the provided date is a trading date
        try:
            position = self.six_trading_calendar.index.get_loc(trading_date)
        except:
            raise ValueError(f"{trading_date} is not a trading date or is out of the given date range.")
        
        # Calculate the new date position by adding the number of trading days
        new_position = position + num_trading_day

        # Retrieve the new trading date from the calendar
        try:
            return self.six_trading_calendar.index[new_position] # Returns the index of the position (cannot use iloc)
        except:
            raise IndexError("The resulting trading date is out of bounds.")

# Example usage of the SIXTradingCalendar class
if __name__ == "__main__":
    # Create an instance of the SIXTradingCalendar class
    trading_calendar = SIXTradingCalendar()

    # Perform various operations using the class instance
    # For example, calculate the number of business days between two dates
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-01-10')
    business_days = trading_calendar.calculate_business_days(start_date, end_date)
    print(f"Number of business days: {business_days}")
    