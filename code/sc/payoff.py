import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

#Initial fixing date: 27/04/23 -> 0
#Final fixing date: 30/07/24 (15 months) ->
#Early redemption dates: look at excel (TODO!!!)

## Return several dfs based on alternate product
## (Original), (No autocall), (No barrier), (No autocall and no barrier)
## Last one is basically trivial since it is completely asset independent, nonetheless nice to consider

## path dfs are length-variable, but should always end on final fixing date
## historical path only to determine barrier hit (not early redemption, since no market afterwards)
def payouts(df_path1, df_path2):
    #init
    df_payouts = pd.DataFrame({'payout': [], 'date': []})
    first_date = df_path1.sort_values('date').iloc[0]['date']
    print(first_date)

    
    #Early redemption, does not yield dividends past called date
    triggerDate = datetime.date.fromisoformat('2024-07-30') #init to final fixing date
    redemptionDate = datetime.date.fromisoformat('2024-08-05') #init to final redemption date
    autocall_dates = [[datetime.date.fromisoformat('2023-11-01'), datetime.date.fromisoformat('2024-01-31'), datetime.date.fromisoformat('2024-04-30')],
                      [datetime.date.fromisoformat('2023-11-06'), datetime.date.fromisoformat('2024-02-05'), datetime.date.fromisoformat('2024-05-06')]]
    for i in range(len(autocall_dates[0])):
        date = autocall_dates[0][i]
        #print(date)
        if df_path1.loc[df_path1['date'] == date].iloc[0]['price'] >= 1 and df_path2.loc[df_path2['date'] == date].iloc[0]['price'] >= 1:
            #print("autocall at {date}!".format(date=date))
            triggerDate = date
            redemptionDate = autocall_dates[1][i]
            #er payout added as final payout later
            #er_payout = pd.DataFrame({'payout': [1000], 'date': [redemptionDate]})
            #df_payouts = pd.concat([df_payouts, er_payout])
            break
    
    #barrier check
    #can and should be optimised for runtime, since barrier hit only needs to be bool
    #current formulation for checking and viz purpose
    barrierHit = False #change me when considering varying path start date
    if min(df_path1['price']) < 0.6:
        barrierHit = True
        barrier1Hit = df_path1.loc[df_path1['price'] < 0.6].sort_values(by='date').iloc[0]
        #print("path1 hit barrier on {date} with price {price}".format(date = barrier1Hit['date'], price = barrier1Hit['price']))
    if min(df_path2['price']) < 0.6:
        barrierHit = True
        barrier2Hit = df_path2.loc[df_path2['price'] < 0.6].sort_values(by='date').iloc[0]
        #print("path2 hit barrier on {date} with price {price}".format(date = barrier2Hit['date'], price = barrier2Hit['price']))
    
    #dividend payment
    div_payment_dates = [datetime.date.fromisoformat('2023-08-07'), datetime.date.fromisoformat('2023-11-06'), datetime.date.fromisoformat('2024-02-05'), datetime.date.fromisoformat('2024-05-06'), datetime.date.fromisoformat('2024-08-05')]
    for date in div_payment_dates:
        if date <= redemptionDate:
            div_payout = pd.DataFrame({'payout': [1000 * 0.02], 'date': [date]})
            df_payouts = pd.concat([df_payouts, div_payout])
    
    #Final redemption
    #if early redemption occured, payout = 1000 regardless of barrierHit
    if barrierHit:
        path1Closing = df_path1.loc[df_path1['date'] == triggerDate].iloc[0]['price']
        path2Closing = df_path2.loc[df_path2['date'] == triggerDate].iloc[0]['price']
        worstPerforming = min(path1Closing, path2Closing, 1)
        final_payout = pd.DataFrame({'payout': [1000 * worstPerforming], 'date': [redemptionDate]})
    else:
        final_payout = pd.DataFrame({'payout': [1000], 'date': [redemptionDate]})
    df_payouts = pd.concat([df_payouts, final_payout])
    return df_payouts
        
#interesting idea: def payoff as payoff_historical + payoff_simulated to save runtime
#pass barrier hit bool from historical to simulated
#actually no, historical payoff not relevant to product price
#since ppl buying the product wont get coupon payment anyways
#the only important thing is barrier hit and early redemption
#early redemption not so impt since product simply becomes untradable so no need to simulate
