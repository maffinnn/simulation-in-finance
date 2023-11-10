import os
print(os.getcwd())
from yq.utils import option
from yq.utils import calendar
import pandas as pd
from yq.scripts import models
from yq.scripts import heston
data = option.read_options_data("lonn_call.csv")
print(data)

# Create an instance of the SIXTradingCalendar class
trading_calendar = calendar.SIXTradingCalendar()

# Perform various operations using the class instance
# For example, calculate the number of business days between two dates
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-01-10')
business_days = trading_calendar.calculate_business_days(start_date, end_date)
print(f"Number of business days: {business_days}")


if __name__ == "__main__":
    params = {
    'data': data,
    'ticker_list': ['LONN.SW', 'SIKA.SW']
    }
    trading_calendar = calendar.SIXTradingCalendar()
    heston = models.PricingModel(params = params)

    bus_date_range = trading_calendar.create_six_trading_dates('2023-08-09', '2023-08-09')
    # print(bus_date_range)
    # print(bus_date_range.index.to_list())
    for product_est_date in bus_date_range.index:
        # print(date, type(date))
        
        try:
            sim_start_date = trading_calendar.add_trading_day(product_est_date, 1)
            sim_data = heston.multi_asset_heston_model(
                sim_start_date=sim_start_date, 
                hist_window=trading_calendar.calculate_business_days(sim_start_date, 
                                                                    cs.FINAL_FIXING_DATE), 
                sim_window=252, h_adjustment=[0, 0])
            # print(sim_data)
        except Exception as e:
            # Log the error with the date that caused it
            raise Exception("Heston has error.")
        
        # sim_data.columns = ['LONN.SW', 'SIKA.SW']
        # sim_data['LONN.SW'] 