from sklearn.linear_model import LinearRegression
import numpy as np

def apply_control_variates(S1T_n, S2T_n, mu1, mu2, payouts_n):
    ratio = 0.5
    N = len(payouts_n)
    reg_len = int(ratio*N)
    S1T_n_reg = S1T_n[:reg_len]
    S1T_n_est = S1T_n[reg_len:]
    S2T_n_reg = S2T_n[:reg_len]
    S2T_n_est = S2T_n[reg_len:]
    ST_n_reg = np.array([S1T_n_reg, S2T_n_reg])
    ST_n_est = np.array([S1T_n_est, S2T_n_est])
    payouts_n_reg = payouts_n[:reg_len]
    payouts_n_est = payouts_n[reg_len:]

    reg = LinearRegression().fit(np.transpose(ST_n_reg), payouts_n_reg)
    coef = reg.coef_
    mu = np.array([[mu1, mu2] for _ in range(N-reg_len)])
    payouts_cv = payouts_n_est + np.dot((mu - np.transpose(ST_n_est)), coef)
    return payouts_cv