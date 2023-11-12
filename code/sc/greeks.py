import sc.payoff as po
import sc.constants as cs

# Simple greeks implementations

#TODO: greeks

#evaluates delta for a single price path
def greeks_gbm(h, df_sim, asset):
    first_sim_date = df_sim.first_valid_index()
    df_historical = po.get_historical_assets(first_sim_date, cs.INITIAL_FIXING_DATE)
    today = df_historical.index[-1]
    barrierHit = po.checkBarrier(df_historical)

    chi_0 = po.rnv(po.payouts(df_sim, barrierHit), today)
    df_plus_h = df_sim.copy(deep = True)
    df_plus_h[asset] = df_plus_h[asset] * (1 + h)
    chi_plus_h = po.rnv(po.payouts(df_plus_h, barrierHit), today)
    df_minus_h = df_sim.copy(deep = True)
    df_minus_h[asset] = df_minus_h[asset] * (1 - h)
    chi_minus_h = po.rnv(po.payouts(df_minus_h, barrierHit), today)

    delta = (chi_plus_h - chi_minus_h) / (2 * h)
    gamma = (chi_plus_h - 2 * chi_0 + chi_minus_h) / (h ** 2)
    return [delta, gamma]