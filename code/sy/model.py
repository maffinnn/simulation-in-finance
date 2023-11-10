from __future__ import annotations
import typing
import datetime
import numpy as np
import pandas as pd

format = '%Y-%m-%d'
class MultiAssetGBM(object):
    def __init__(self, data: pd.DataFrame, params: typing.Dict):
        self.data = data
        self.name = params.get('name', None)
        self.maturity_date =  params.get('maturity_date')
        self.dt = 1/252
        self.asset_names = params.get('asset_names')
        self.Nassets = len(self.asset_names)
        self.window_size = params.get('window_size')
        self.interest_rate_model = params.get('interest_rate_model')
        self.volatiliy_model = params.get('volatiliy_model')
        
    def __log_return(self, current_date: str)->pd.DataFrame:
        window_end = datetime.datetime.strptime(current_date, format) - datetime.timedelta(days=1)
        window_start, window_end = (window_end - datetime.timedelta(days=self.window_size)).strftime(format), window_end.strftime(format)
        if self.log_return is None:
            self.log_return = np.log(self.data) - np.log(self.data.shift(1))
        return self.log_return.loc[window_start: window_end]
    
    def fit(self, current_date: str)->MultiAssetGBM:
        window_end = (datetime.datetime.strptime(current_date, format) - datetime.timedelta(days=1)).strftime(format)
        self.S0 = self.data.loc[window_end].to_numpy()
        self.windowed_log_return = self.__log_return(current_date)
        self.T = np.busday_count(current_date, self.maturity_date) # time to maturity
        cov = self.windowed_log_return.cov()
        self.sigma = np.linalg.cholesky(cov) * self.dt
        self.mu = self.interest_rate_model.fit(current_date).generate_path().loc[current_date] if self.interest_rate_model else self.windowed_log_return.mean().to_numpy().reshape(self.Nassets, 1)
        self.var = cov.to_numpy().diagonal().reshape(self.Nassets, 1)
        return self    
    
    def get_random_variables(self)->np.ndarray:
        if self.rv is None:
            self.rv = np.random.normal(0, np.sqrt(self.dt), size=(self.Nassets, self.T))
        return self.rv
    
    def generate_path(self)->np.ndarray:
        St = np.exp(
            (self.mu - self.var/2) * self.dt + np.matmul(self.sigma * self.rv)
        )
        St = np.vstack([np.ones(self.Nassets), St])
        St = self.S0 * St.cumprod(axis=0)
        return St
        
        
def checkBarrier(df_path1, df_path2):
    return min(df_path1['price']) < 0.6 or min(df_path2['price']) < 0.6

def genGBM(s_0, t_0, mu, sigma, delta_t, n):
    df_path = pd.DataFrame(data={'price': [s_0], 'date': [t_0]})
    curDate = t_0
    curPrice = s_0
    for i in range(n):
        curDate += datetime.timedelta(days=1)
        z = np.random.normal()
        drift = (mu - np.power(sigma, 2) / 2) * delta_t
        diffusion = np.power(delta_t, 0.5) * sigma
        curPrice *= np.exp(drift + diffusion * z)
        newRow = pd.DataFrame(data={'price': [curPrice], 'date': [curDate]})
        df_path = pd.concat([df_path, newRow])
    return df_path

#placeholder interest rate lookup formula
def ir_lookup(date):
    r = 0.05
    return r

#return discounted payoff from df of payouts
#constant riskfree rate r, TODO business day translation
def rnv(df_payouts, today, r):
    #ttp = time to payout
    df_payouts['ttp'] = [tdelta.days for tdelta in (df_payouts['date'] - today).to_list()]
    df_payouts['disc_price'] = df_payouts['payout'] * np.exp(-r * df_payouts['ttp'] / 365)
    return sum(df_payouts['disc_price'])


def future_payouts(df_path1, df_path2, barrierHit):
    #init
    df_payouts = pd.DataFrame({'payout': [], 'date': []})
    first_date = df_path1.sort_values('date').iloc[0]['date']
    
    #Early redemption, does not yield dividends past called date
    triggerDate = datetime.date.fromisoformat('2024-07-30') #init to final fixing date
    redemptionDate = datetime.date.fromisoformat('2024-08-05') #init to final redemption date
    autocall_dates = [[datetime.date.fromisoformat('2023-11-01'), datetime.date.fromisoformat('2024-01-31'), datetime.date.fromisoformat('2024-04-30')],
                      [datetime.date.fromisoformat('2023-11-06'), datetime.date.fromisoformat('2024-02-05'), datetime.date.fromisoformat('2024-05-06')]]
    for i in range(len(autocall_dates[0])):
        date = autocall_dates[0][i]
        #only check dates after today
        if date >= first_date:
            #if both assets are above reference level on 'date'
            if df_path1.loc[df_path1['date'] == date].iloc[0]['price'] >= 1 and df_path2.loc[df_path2['date'] == date].iloc[0]['price'] >= 1:
                #print("autocall at {date}!".format(date=date))
                triggerDate = date
                redemptionDate = autocall_dates[1][i]
                break
    
    #barrier check
    if not barrierHit:
        barrierHit = checkBarrier(df_path1, df_path2)
    
    #dividend payment
    div_payment_dates = [datetime.date.fromisoformat('2023-08-07'), datetime.date.fromisoformat('2023-11-06'), datetime.date.fromisoformat('2024-02-05'), datetime.date.fromisoformat('2024-05-06'), datetime.date.fromisoformat('2024-08-05')]
    for date in div_payment_dates:
        #div payments after today
        if date > first_date:
            if date <= redemptionDate:
                div_payout = pd.DataFrame({'payout': [1000 * 0.02], 'date': [date]})
                df_payouts = pd.concat([df_payouts, div_payout])
    
    #Final redemption
    #if early redemption occured, payout = 1000 regardless of barrierHit since both underlyings would be > 1 anyways
    if barrierHit:
        path1Closing = df_path1.loc[df_path1['date'] == triggerDate].iloc[0]['price']
        path2Closing = df_path2.loc[df_path2['date'] == triggerDate].iloc[0]['price']
        worstPerforming = min(path1Closing, path2Closing, 1)
        final_payout = pd.DataFrame({'payout': [1000 * worstPerforming], 'date': [redemptionDate]})
    else:
        final_payout = pd.DataFrame({'payout': [1000], 'date': [redemptionDate]})
    df_payouts = pd.concat([df_payouts, final_payout])
    return df_payouts
