import logging
import pandas as pd
from pathlib import Path
import os
import numpy as np

from sc import constants as cs
from yq.utils import path as yq_path
import sy.interest_rate as syir

logger_yq = logging.getLogger('yq')

# ===============================================================================================
# Generic Helper Functions
# ===============================================================================================

def average(arr):
    return sum(arr) / len(arr)

# ===============================================================================================
# Historical
# ===============================================================================================

## price path df structure
## (index date), (stock1 price LONN.SE), (stock2 price SIKA.SE)

# get df of actual product price up to and including today
# similar format as simulated price paths, column 'Price'
def get_product_price(today):
    cur_dir = Path(__file__).parent
    root_dir = yq_path.get_root_dir(cur_dir=cur_dir)
    json_file_path = root_dir.joinpath('code', 'sc', 'product_price.json')
    # logger_yq.info(f"The json file path for product price is: {json_file_path}")
    df_product = pd.read_json(json_file_path).rename(columns = {'date': 'Date', 'value': 'Price'})
    df_product['Date'] = pd.to_datetime(df_product['Date'])
    df_product['Price'] = df_product['Price'] / 100 * cs.DENOMINATION
    df_product = df_product.loc[df_product['Date'] <= today].set_index('Date').sort_index()
    return df_product

# get historical underlying prices up to but not including first_sim_date
# same format as simulated price paths
def get_historical_assets(first_sim_date, start_date = pd.Timestamp('2019-01-01')):
    first = True
    for asset in cs.ASSET_NAMES:
        cur_dir = Path(__file__).parent
        root_dir = yq_path.get_root_dir(cur_dir=cur_dir)
        json_file_path = root_dir.joinpath('code', 'sc', f'{asset}.json')
        # logger_yq.info(f"The historical asset json file path is: {json_file_path}")
        df_asset = pd.read_json(json_file_path).drop(['high', 'low', 'close', 'open'], axis = 1)
        df_asset = df_asset.rename(columns = {'date': 'Date', 'value': asset})
        df_asset['Date'] = pd.to_datetime(df_asset['Date'])
        df_asset = df_asset.loc[df_asset['Date'] < first_sim_date].loc[df_asset['Date'] >= start_date].set_index('Date').sort_index()
        if first:
            df_historical = df_asset.copy(deep = True)
            first = False
        else:
            df_historical = df_historical.merge(df_asset, on = 'Date')
    return df_historical

# Dump all historical data
# same format as simulated price paths
def get_historical_assets_all():
    first = True
    for asset in cs.ASSET_NAMES:
        cur_dir = Path(__file__).parent
        root_dir = yq_path.get_root_dir(cur_dir=cur_dir)
        json_file_path = root_dir.joinpath('code', 'sc', f'{asset}.json')
        # logger_yq.info(f"The json file path is: {json_file_path}")
        df_asset = pd.read_json(json_file_path).drop(['high', 'low', 'close', 'open'], axis = 1)
        df_asset = df_asset.rename(columns = {'date': 'Date', 'value': asset})
        df_asset['Date'] = pd.to_datetime(df_asset['Date'])
        df_asset = df_asset.set_index('Date').sort_index()
        if first:
            df_historical = df_asset.copy(deep = True)
            first = False
        else:
            df_historical = df_historical.merge(df_asset, on = 'Date')
    return df_historical

# Helper function to check barrier hit in some historical price df
def checkBarrier(df_historical):
    barrierHit = False
    for asset in cs.ASSET_NAMES:
        initial_price = cs.INITIAL_LEVELS[asset]
        barrierHit = barrierHit or min(df_historical[asset]) < cs.BARRIER * initial_price
    return barrierHit


# ===============================================================================================
# Payoff Functions
# ===============================================================================================

# path dfs are length-variable, but should always end on final fixing date
# historical path only to determine barrier hit (not early redemption, since no market afterwards)
# this fn takes in only simulated portion!!! interprets first row as 'today'
def payouts(df_sim, barrierHit):
    #init
    df_payouts = pd.DataFrame()
    first_date = df_sim.first_valid_index()
    
    #Early redemption, does not yield dividends past called date
    trigger_date = cs.FINAL_FIXING_DATE #init to final fixing date
    redemption_date = cs.REDEMPTION_DATE #init to final redemption date
    for date in cs.EARLY_REDEMPTION_OBSERVATION_DATES:
        if date >= first_date:
            autocall_hit = True
            for asset in cs.ASSET_NAMES:
                autocall_hit = autocall_hit and df_sim.loc[date][asset] >= cs.INITIAL_LEVELS[asset] * cs.EARLY_REDEMPTION_LEVEL
            if autocall_hit:
                trigger_date = date
                redemption_date = cs.EARLY_REDEMPTION_DATES[date]
                break

    #barrier check
    if not barrierHit:
        #check barrier for sim path
        for asset in cs.ASSET_NAMES:
            if min(df_sim[asset]) < cs.BARRIER * cs.INITIAL_LEVELS[asset]:
                barrierHit = True
                break
    
    #dividend payment
    for date in cs.COUPON_PAYMENT_DATES:
        if date <= redemption_date and date > first_date:
            div_payout = pd.DataFrame({'Payout': [cs.COUPON_PAYOUT], 'Date': [date]})
            df_payouts = pd.concat([df_payouts, div_payout], axis = 0)
    
    #Final redemption
    #if early redemption occured, payout = 1000 regardless of barrierHit
    if barrierHit:
        worst_performing = cs.DENOMINATION
        for asset in cs.ASSET_NAMES:
            final_price = df_sim.loc[trigger_date][asset] * cs.CONVERSION_RATIOS[asset] #this will be > cs.DENOMINATION if autocall occurred
            worst_performing = min(worst_performing, final_price)
        final_payout = pd.DataFrame({'Payout': [worst_performing], 'Date': [redemption_date]})
    else:
        final_payout = pd.DataFrame({'Payout': [cs.DENOMINATION], 'Date': [redemption_date]})
    df_payouts = pd.concat([df_payouts, final_payout], axis = 0)
    return df_payouts
        
## Alt product payouts

# No autocall, yes barrier
# Price diff uncertain, not autocalling could lead to barrier hit in the future
def payouts_no_autocall(df_sim, barrierHit):
    #init
    df_payouts = pd.DataFrame()
    first_date = df_sim.first_valid_index()
    
    trigger_date = cs.FINAL_FIXING_DATE #init to final fixing date
    redemption_date = cs.REDEMPTION_DATE #init to final redemption date

    #barrier check
    if not barrierHit:
        #check barrier for sim path
        for asset in cs.ASSET_NAMES:
            if min(df_sim[asset]) < cs.BARRIER * cs.INITIAL_LEVELS[asset]:
                barrierHit = True
                break
    
    #dividend payment
    for date in cs.COUPON_PAYMENT_DATES:
        if date <= redemption_date and date > first_date:
            div_payout = pd.DataFrame({'Payout': [cs.COUPON_PAYOUT], 'Date': [date]})
            df_payouts = pd.concat([df_payouts, div_payout], axis = 0)
    
    #Final redemption
    #if early redemption occured, payout = 1000 regardless of barrierHit
    if barrierHit:
        worst_performing = cs.DENOMINATION
        for asset in cs.ASSET_NAMES:
            final_price = df_sim.loc[trigger_date][asset] * cs.CONVERSION_RATIOS[asset] #this will be > cs.DENOMINATION if autocall occurred
            worst_performing = min(worst_performing, final_price)
        final_payout = pd.DataFrame({'Payout': [worst_performing], 'Date': [redemption_date]})
    else:
        final_payout = pd.DataFrame({'Payout': [cs.DENOMINATION], 'Date': [redemption_date]})
    df_payouts = pd.concat([df_payouts, final_payout], axis = 0)
    return df_payouts

# Basically a bond with a risk of early redemption
# expect higher price wrt original
# barrierHit param not used but left in to maintain consistency
def payouts_no_barrier(df_sim, barrierHit):
    #init
    df_payouts = pd.DataFrame()
    first_date = df_sim.first_valid_index()
    
    #Early redemption, does not yield dividends past called date
    trigger_date = cs.FINAL_FIXING_DATE #init to final fixing date
    redemption_date = cs.REDEMPTION_DATE #init to final redemption date
    for date in cs.EARLY_REDEMPTION_OBSERVATION_DATES:
        if date >= first_date:
            autocall_hit = True
            for asset in cs.ASSET_NAMES:
                autocall_hit = autocall_hit and df_sim.loc[date][asset] >= cs.INITIAL_LEVELS[asset] * cs.EARLY_REDEMPTION_LEVEL
            if autocall_hit:
                trigger_date = date
                redemption_date = cs.EARLY_REDEMPTION_DATES[date]
                break
    
    #dividend payment
    for date in cs.COUPON_PAYMENT_DATES:
        if date <= redemption_date and date > first_date:
            div_payout = pd.DataFrame({'Payout': [cs.COUPON_PAYOUT], 'Date': [date]})
            df_payouts = pd.concat([df_payouts, div_payout], axis = 0)
    
    #Final redemption
    final_payout = pd.DataFrame({'Payout': [cs.DENOMINATION], 'Date': [redemption_date]})
    df_payouts = pd.concat([df_payouts, final_payout], axis = 0)
    return df_payouts

# Basically a bond
# expect higher price than everything else
def payouts_no_barrier_no_autocall(df_sim, barrierHit):
    #init
    df_payouts = pd.DataFrame()
    first_date = df_sim.first_valid_index()
    
    trigger_date = cs.FINAL_FIXING_DATE #init to final fixing date
    redemption_date = cs.REDEMPTION_DATE #init to final redemption date

    #dividend payment
    for date in cs.COUPON_PAYMENT_DATES:
        if date <= redemption_date and date > first_date:
            div_payout = pd.DataFrame({'Payout': [cs.COUPON_PAYOUT], 'Date': [date]})
            df_payouts = pd.concat([df_payouts, div_payout], axis = 0)
    
    #Final redemption
    final_payout = pd.DataFrame({'Payout': [cs.DENOMINATION], 'Date': [redemption_date]})
    df_payouts = pd.concat([df_payouts, final_payout], axis = 0)
    return df_payouts


# ===============================================================================================
# RNV
# ===============================================================================================

def populate_bond_table_sc(today, bond_price):
    bond_table = pd.DataFrame(index=pd.date_range(today, cs.REDEMPTION_DATE), columns=['Price'])
    bond_table.index.name = 'Date'
    X = [syir.get_period(col) for col in bond_price.columns]
    Y = bond_price.loc[today].to_list()
    for date in bond_table.index:
        #tdelta = (cs.FINAL_FIXING_DATE - date).days/365
        tdelta = (date - today).days / 365
        # for date in bond_price.index:
        interpolated_y = np.interp(tdelta,X,Y)
        bond_table.loc[date]['Price'] = interpolated_y
    return bond_table

def create_bond_table_sc(today):
    cur_dir = Path(__file__).parent
    root_dir = yq_path.get_root_dir(cur_dir=cur_dir)
    tar_dir = root_dir.joinpath('data', 'bond')
    bond_yield = None
    for file in os.listdir(tar_dir):
        df = pd.read_csv(os.path.join(tar_dir, file))[['Date','Price']]
        df.rename(columns={'Price':file.split(' ')[1]}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'],format='%m/%d/%Y')
        df = df.set_index('Date').iloc[::-1]
        if bond_yield is None:
            bond_yield = df
        else:
            bond_yield = pd.concat([bond_yield, df], axis=1)
    bond_yield = bond_yield.interpolate()
    bond_yield = bond_yield.reindex(sorted(bond_yield.columns, key=lambda x: syir.get_period(x)), axis=1)
    bond_price = pd.DataFrame(index=bond_yield.index)
    for col in bond_yield.columns:
        bond_price[col] = bond_yield[col].apply(lambda x: np.exp(-x/100*syir.get_period(col)))
    #print(bond_price)
    #bond_table = syir.populate_bond_table(bond_price, today, cs.FINAL_FIXING_DATE)
    bond_table = populate_bond_table_sc(today, bond_price)
    return bond_table

def rnv_single(df_payouts, today):
    bond_table = create_bond_table_sc(today)
    return np.sum(bond_table.loc[df_payouts['Date']]['Price'].to_numpy() * df_payouts['Payout'].to_numpy())

def rnv_multiple(df_payouts_arr, today):
    bond_table = create_bond_table_sc(today)
    rnv_arr = []
    for df_payouts in df_payouts_arr:
        rnv_arr.append(np.sum(bond_table.loc[df_payouts['Date']]['Price'].to_numpy() * df_payouts['Payout'].to_numpy()))
    return rnv_arr

def rnv_multiple_bond_table(df_payouts, bond_table):
    return np.sum(bond_table.loc[df_payouts['Date']]['Price'].to_numpy() * df_payouts['Payout'].to_numpy())

# ===============================================================================================
# Run Pricing (Abstracted)
# ===============================================================================================

# Put in a df of simulated price paths and get back RNV price
def pricing_single(df_sim):
    first_sim_date = df_sim.first_valid_index()
    df_historical = get_historical_assets(first_sim_date, cs.INITIAL_FIXING_DATE)
    barrierHit = checkBarrier(df_historical)
    df_payouts = payouts(df_sim, barrierHit)
    price = rnv_single(df_payouts, df_historical.index[-1])
    return price

# put in an array of many dfs, get back an array of their prices
def pricing_multiple(df_sim_array):
    first_sim_date = df_sim_array[0].first_valid_index()
    df_historical = get_historical_assets(first_sim_date, cs.INITIAL_FIXING_DATE)
    today = df_historical.index[-1]
    barrierHit = checkBarrier(df_historical)
    df_payouts_arr = []
    for df_sim in df_sim_array:
        df_payouts = payouts(df_sim, barrierHit)
        df_payouts_arr.append(df_payouts)
    rnv_arr = rnv_multiple(df_payouts_arr, today)
    return rnv_arr


# ===============================================================================================
# Sensitivity Analysis
# ===============================================================================================

# Look in greeks.py for simple implementations!
# Watch this space for simultaneous payoff and greek calculations.

# returns 3 payout dfs in a list, [0, +, -]
def payouts_h(df_sim, barrierHit, h, asset_h):
    #init
    df_payouts_arr = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()] #[0, +, -]
    factor = [1, 1 + h, 1 - h]
    first_date = df_sim.first_valid_index()
    
    #Early redemption, does not yield dividends past called date
    trigger_date = [cs.FINAL_FIXING_DATE] * 3 #init to final fixing date
    redemption_date = [cs.REDEMPTION_DATE] * 3 #init to final redemption date

    #for each path
    for i in range(3):
        #at each autocall date
        for date in cs.EARLY_REDEMPTION_OBSERVATION_DATES:
            #after today
            if date >= first_date:
                autocall_hit = True
                #iterate through each asset to check for autocall
                for asset in cs.ASSET_NAMES:
                    if asset == asset_h:
                        autocall_hit = autocall_hit and (df_sim.loc[date][asset] * factor[i] >= cs.INITIAL_LEVELS[asset] * cs.EARLY_REDEMPTION_LEVEL)
                    else:
                        autocall_hit = autocall_hit and (df_sim.loc[date][asset] >= cs.INITIAL_LEVELS[asset] * cs.EARLY_REDEMPTION_LEVEL)
                if autocall_hit:
                    trigger_date[i] = date
                    redemption_date[i] = cs.EARLY_REDEMPTION_DATES[date]
                    break #dont check subsequent autocall dates

    #barrier check
    barrier_arr = [barrierHit] * 3
    if not barrierHit:
        #check barrier for simulated paths
        for asset in cs.ASSET_NAMES:
            if asset != asset_h:
                if min(df_sim[asset]) < cs.BARRIER * cs.INITIAL_LEVELS[asset]:
                    barrier_arr = [True] * 3
                    break
            else:
                asset_h_min = min(df_sim[asset])
                for i in range(3):
                    if asset_h_min * factor[i] < cs.BARRIER * cs.INITIAL_LEVELS[asset]:
                        barrier_arr[i] = True
    
    #dividend payment
    for i in range(3):
        for date in cs.COUPON_PAYMENT_DATES:
            if date <= redemption_date[i] and date > first_date:
                div_payout = pd.DataFrame({'Payout': [cs.COUPON_PAYOUT], 'Date': [date]})
                df_payouts_arr[i] = pd.concat([df_payouts_arr[i], div_payout], axis = 0)
    
    #Final redemption
    #if early redemption occured, payout = 1000 regardless of barrierHit
    for i in range(3):
        if barrier_arr[i]:
            worst_performing = cs.DENOMINATION
            for asset in cs.ASSET_NAMES:
                if asset != asset_h:
                    final_price = df_sim.loc[trigger_date[i]][asset] * cs.CONVERSION_RATIOS[asset] #this will be > cs.DENOMINATION if autocall occurred
                else:
                    final_price = df_sim.loc[trigger_date[i]][asset] * factor[i] * cs.CONVERSION_RATIOS[asset] #this will be > cs.DENOMINATION if autocall occurred
                worst_performing = min(worst_performing, final_price)
            final_payout = pd.DataFrame({'Payout': [worst_performing], 'Date': [redemption_date[i]]})
        else:
            final_payout = pd.DataFrame({'Payout': [cs.DENOMINATION], 'Date': [redemption_date[i]]})
        df_payouts_arr[i] = pd.concat([df_payouts_arr[i], final_payout], axis = 0)
    return df_payouts_arr

def delta(price_arr, h):
    return (price_arr[1] - price_arr[2]) / (2 * h)

def gamma(price_arr, h):
    return (price_arr[1] - 2 * price_arr[0] + price_arr[2]) / (h ** 2)

# returns list of [price, greeks]
# greeks is a dict with keys 'asset name' and values [delta, gamma]
def pricing_with_greeks_single(df_sim, h):
    first_sim_date = df_sim.first_valid_index()
    df_historical = get_historical_assets(first_sim_date, cs.INITIAL_FIXING_DATE)
    today = df_historical.index[-1]
    barrierHit = checkBarrier(df_historical)

    price = -1
    greeks = {}
    for asset in cs.ASSET_NAMES:
        df_payouts_arr = payouts_h(df_sim, barrierHit, h, asset)
        rnv_arr = rnv_multiple(df_payouts_arr, today)
        price = rnv_arr[0]
        greeks[asset] = [delta(rnv_arr, h), gamma(rnv_arr, h)]
    return [price, greeks]

# returns array of lists [price, greeks]
# greeks is a dict with keys 'asset name' and values [delta, gamma]
def pricing_with_greeks_multiple(df_sim_arr, h):
    results_arr = []
    
    first_sim_date = df_sim_arr[0].first_valid_index()
    df_historical = get_historical_assets(first_sim_date, cs.INITIAL_FIXING_DATE)
    today = df_historical.index[-1]
    barrierHit = checkBarrier(df_historical)

    bond_table = create_bond_table_sc(today)

    for df in df_sim_arr:
        price = -1
        greeks = {}
        for asset in cs.ASSET_NAMES:
            paths_payouts_arr = payouts_h(df, barrierHit, h, asset)
            rnv_arr = []
            for path_payout in paths_payouts_arr:
                rnv_arr.append(rnv_multiple_bond_table(path_payout, bond_table))
            price = rnv_arr[0]
            greeks[asset] = [delta(rnv_arr, h), gamma(rnv_arr, h)]
        results_arr.append([price, greeks])
    return results_arr
