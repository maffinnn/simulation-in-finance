import pandas as pd

# Stock tickers
ASSET_NAMES = [
    'LONN.SE',
    'SIKA.SE'
]

INITIAL_LEVELS = {
    'LONN.SE': 549.60,
    'SIKA.SE': 240.40
}
CONVERSION_RATIOS = {
    'LONN.SE': 1.8195,
    'SIKA.SE': 4.1597
}

# Information on factsheet
INITIAL_FIXING_DATE = pd.Timestamp("2023-04-27")
PAYMENT_DATE = pd.Timestamp("2023-05-05")
FINAL_FIXING_DATE = pd.Timestamp("2024-07-30")
REDEMPTION_DATE = pd.Timestamp("2024-08-05") # Subject to early redemption

# If all Reference Shares close at or above their Early Redemption Levels on any Early Redemption Observation Date
EARLY_REDEMPTION_LEVEL = 1.00  # 100% of the Initial Level
EARLY_REDEMPTION_OBSERVATION_FREQUENCY = "quarterly"

# CURRENCY = "CHF"
DENOMINATION = 1000  # CHF 1,000
ISSUE_PRICE_PERCENTAGE = 1.00  # 100%

# Simulation constants
INITIAL_PROD_PRICING_DATE = pd.Timestamp('2023-08-09')
FINAL_PROD_PRICING_DATE = pd.Timestamp('2023-11-09')
PRICING_WINDOW = 67
INTEREST_RATE = 1.750 / 100
# SIMULATION_START_DATE = next day from the date of the product price estimation

# Historical stock prices to fetch
HISTORICAL_START_DATE = pd.Timestamp('2022-08-09')
# HISTORICAL_END_DATE = today

DENOMINATION = 1000.0
BARRIER = 0.6
COUPON_RATE = 0.08 / 4
COUPON_PAYOUT = COUPON_RATE * DENOMINATION

COUPON_PAYMENT_DATES = [
    pd.Timestamp('2023-08-07'),
    pd.Timestamp('2023-11-06'),
    pd.Timestamp('2024-02-05'),
    pd.Timestamp('2024-05-06'),
    pd.Timestamp('2024-08-05'),
]

EARLY_REDEMPTION_OBSERVATION_DATES = [
    pd.Timestamp('2023-11-01'),
    pd.Timestamp('2024-01-31'),
    pd.Timestamp('2024-04-30'),
]

EARLY_REDEMPTION_DATES = {
    pd.Timestamp('2023-11-01'): pd.Timestamp('2023-11-06'),
    pd.Timestamp('2024-01-31'): pd.Timestamp('2024-02-05'),
    pd.Timestamp('2024-04-30'): pd.Timestamp('2024-05-06'),
}

# https://www.six-group.com/en/products-services/the-swiss-stock-exchange/market-data/news-tools/trading-currency-holiday-calendar.html#/
# Double checked with Sze Chong's product dates
SIX_HOLIDAY_DATES = [
    pd.Timestamp('2023-01-02'),  # Berchtholdstag
    pd.Timestamp('2023-04-07'),  # Good Friday
    pd.Timestamp('2023-04-10'),  # Easter Monday
    pd.Timestamp('2023-05-01'),  # Labour Day
    pd.Timestamp('2023-05-18'),  # Ascension Day
    pd.Timestamp('2023-05-29'),  # Whitmonday
    pd.Timestamp('2023-08-01'),  # National Day
    pd.Timestamp('2023-12-25'),  # Christmas Day
    pd.Timestamp('2023-12-26'),  # St. Stephen's Day
    pd.Timestamp('2024-01-01'),  # New Year's Day
    pd.Timestamp('2024-01-02'),  # Berchtholdstag
    pd.Timestamp('2024-03-29'),  # Good Friday
    pd.Timestamp('2024-04-01'),  # Easter Monday
    pd.Timestamp('2024-05-01'),  # Labour Day
    pd.Timestamp('2024-05-09'),  # Ascension Day
    pd.Timestamp('2024-05-20'),  # Whitmonday
    pd.Timestamp('2024-08-01'),  # National Day
    pd.Timestamp('2024-12-24'),  # Christmas Eve
    pd.Timestamp('2024-12-25'),  # Christmas
    pd.Timestamp('2024-12-26'),  # St. Stephen's Day
    pd.Timestamp('2024-12-31'),  # New Year's Eve
]
