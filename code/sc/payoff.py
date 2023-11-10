import pandas as pd
import numpy as np
import constants as cs

#price path df structure
#(index date), (stock1 price LONN.SE), (stock2 price SIKA.SE)

def checkBarrier(df_historical):
    barrierHit = False
    for i in range(len(cs.ASSET_NAMES)):
        cur_asset = cs.ASSET_NAMES[i]
        initial_price = df_historical.iloc[0][cur_asset]
        barrierHit = barrierHit or min(df_historical[cur_asset]) < cs.BARRIER * initial_price
    return barrierHit

#Initial fixing date: 27/04/23 -> 0
#Final fixing date: 30/07/24 (15 months) ->
#Early redemption dates: look at excel (TODO!!!)

## Return several dfs based on alternate product
## (Original), (No autocall), (No barrier), (No autocall and no barrier)
## Last one is basically trivial since it is completely asset independent, nonetheless nice to consider

## path dfs are length-variable, but should always end on final fixing date
## historical path only to determine barrier hit (not early redemption, since no market afterwards)
## this fn takes in only simulated portion!!! interprets first row as 'today'
def payouts(df_sim, barrierHit):
    #init
    df_payouts = pd.DataFrame({'payout': [], 'date': []})
    first_date = df_sim.index[0]
    #print(first_date)
    
    #Early redemption, does not yield dividends past called date
    trigger_date = cs.FINAL_FIXING_DATE #init to final fixing date
    redemption_date = cs.REDEMPTION_DATE #init to final redemption date
    for date in cs.EARLY_REDEMPTION_OBSERVATION_DATES:
        autocall_hit = True
        for asset in cs.ASSET_NAMES:
            autocall_hit = autocall_hit and df_sim.loc[date][asset] >= cs.INITIAL_LEVELS[asset] * cs.EARLY_REDEMPTION_LEVEL
        if autocall_hit:
            trigger_date = date
            redemption_date = cs.EARLY_REDEMPTION_DATES[date]
            break
    
    #barrier check
    #can and should be optimised for runtime, since barrier hit only needs to be bool
    #current formulation for checking and viz purpose
    if not barrierHit:
        #check barrier for sim path
        for asset in cs.ASSET_NAMES:
            if min(df_sim[asset]) < cs.BARRIER * cs.INITIAL_LEVELS[asset]:
                barrierHit = True
                break
    
    #dividend payment
    for date in cs.COUPON_PAYMENT_DATES:
        if date <= redemption_date and date > first_date:
            div_payout = pd.DataFrame({'payout': [cs.COUPON_PAYOUT], 'date': [date]})
            df_payouts = pd.concat([df_payouts, div_payout])
    
    #Final redemption
    #if early redemption occured, payout = 1000 regardless of barrierHit
    if barrierHit:
        worst_performing = cs.DENOMINATION
        for asset in cs.ASSET_NAMES:
            final_price = df_sim[trigger_date][asset] #this will be > cs.DENOMINATION if autocall occurred
            worst_performing = min(worst_performing, final_price)
        final_payout = pd.DataFrame({'payout': [worst_performing], 'date': [redemption_date]})
    else:
        final_payout = pd.DataFrame({'payout': [cs.DENOMINATION], 'date': [redemption_date]})
    df_payouts = pd.concat([df_payouts, final_payout])
    return df_payouts
        
#interesting idea: def payoff as payoff_historical + payoff_simulated to save runtime
#pass barrier hit bool from historical to simulated
#actually no, historical payoff not relevant to product price
#since ppl buying the product wont get coupon payment anyways
#the only important thing is barrier hit and early redemption
#early redemption not so impt since product simply becomes untradable so no need to simulate


def rnv(df_payouts, today):
    return 10000

def pricing(df_historical, df_sim):
    barrierHit = checkBarrier(df_historical) #can be optimised to not check historical for every sim path
    df_payouts = payouts(df_sim, barrierHit)
    price = rnv(df_payouts, df_sim.index[0])
    return price