import pandas as pd
from sc import constants as cs


# ===============================================================================================
# Historical
# ===============================================================================================

## price path df structure
## (index date), (stock1 price LONN.SE), (stock2 price SIKA.SE)

# get df of actual product price up to and including today
# similar format as simulated price paths, column 'Price'
def get_product_price(today):
    df_product = pd.read_json('sc/product_price.json').rename(columns = {'date': 'Date', 'value': 'Price'})
    df_product['Date'] = pd.to_datetime(df_product['Date'])
    df_product['Price'] = df_product['Price'] / 100 * cs.DENOMINATION
    df_product = df_product.loc[df_product['Date'] <= today].set_index('Date').sort_index()
    return df_product

# get historical underlying prices up to but not including first_sim_date
# same format as simulated price paths
def get_historical_assets(first_sim_date, start_date = pd.Timestamp('2019-01-01')):
    first = True
    for asset in cs.ASSET_NAMES:
        df_asset = pd.read_json('sc/' + asset + '.json').drop(['high', 'low', 'close', 'open'], axis = 1)
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
        df_asset = pd.read_json('sc/' + asset + '.json').drop(['high', 'low', 'close', 'open'], axis = 1)
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
            final_price = df_sim.loc[trigger_date][asset] #this will be > cs.DENOMINATION if autocall occurred
            worst_performing = min(worst_performing, final_price)
        final_payout = pd.DataFrame({'Payout': [worst_performing], 'Date': [redemption_date]})
    else:
        final_payout = pd.DataFrame({'Payout': [cs.DENOMINATION], 'Date': [redemption_date]})
    df_payouts = pd.concat([df_payouts, final_payout], axis = 0).set_index('Date')
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
            final_price = df_sim.loc[trigger_date][asset] #this will be > cs.DENOMINATION if autocall occurred
            worst_performing = min(worst_performing, final_price)
        final_payout = pd.DataFrame({'Payout': [worst_performing], 'Date': [redemption_date]})
    else:
        final_payout = pd.DataFrame({'Payout': [cs.DENOMINATION], 'Date': [redemption_date]})
    df_payouts = pd.concat([df_payouts, final_payout], axis = 0).set_index('Date')
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
    df_payouts = pd.concat([df_payouts, final_payout], axis = 0).set_index('Date')
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
    df_payouts = pd.concat([df_payouts, final_payout], axis = 0).set_index('Date')
    return df_payouts


# ===============================================================================================
# RNV
# ===============================================================================================

def rnv(df_payouts, today):
    #TODO: refer to interest rates/bond prices and do discounting
    return sum(df_payouts['Payout'])


# ===============================================================================================
# Run Pricing (Abstracted)
# ===============================================================================================

# Put in a df of simulated price paths and get back RNV price
def pricing_single(df_sim):
    first_sim_date = df_sim.first_valid_index()
    df_historical = get_historical_assets(first_sim_date, cs.INITIAL_FIXING_DATE)
    barrierHit = checkBarrier(df_historical)
    df_payouts = payouts(df_sim, barrierHit)
    price = rnv(df_payouts, df_historical.index[-1])
    return price

# put in an array of many dfs, get back an array of their prices
def pricing_multiple(df_sim_array):
    first_sim_date = df_sim_array[0].first_valid_index()
    df_historical = get_historical_assets(first_sim_date, cs.INITIAL_FIXING_DATE)
    barrierHit = checkBarrier(df_historical)
    price = []
    for df_sim in df_sim_array:
        df_payouts = payouts(df_sim, barrierHit)
        print(df_payouts)
        price.append(rnv(df_payouts, df_historical.index[-1]))
        print(price)
    return price


# ===============================================================================================
# Sensitivity Analysis
# ===============================================================================================

#TODO: greeks

#evaluates delta for a single price path
def delta_single_gbm(h, df_historical, df_sim, asset):
    initial_price = df_historical.iloc[-1][asset]
    df_plus_h = df_sim
    df_plus_h[asset] = df_plus_h[asset]
    return 1000