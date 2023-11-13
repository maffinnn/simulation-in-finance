import pandas as pd
import numpy as np
import datetime

def apply_empirical_martingale_correction(data, df_path_n):
    r = 1.75/100
    N = len(df_path_n)
    df_path = df_path_n[0]
    t_0 = df_path.index[0] - datetime.timedelta(1)
    S0 = data.loc[t_0]
    Z = [pd.DataFrame(index=df_path.index, columns=df_path.columns) for _ in range(N+1)]
    corrected_path = [pd.DataFrame(index=df_path.index, columns=df_path.columns) for _ in range(N+1)]
    for col in df_path.columns:
        for j in range(len(df_path.index)):
            for i in range(1, N+1):
                if j == 0:
                    Z[i].iloc[j][col] = S0[col] * df_path_n[i-1].iloc[j][col]/df_path_n[i-1].iloc[j-1][col]
                else:
                    Z[i].iloc[j][col] = corrected_path[i].iloc[j-1][col] * df_path_n[i-1].iloc[j][col]/df_path_n[i-1].iloc[j-1][col]
            date = df_path.index[j]
            t_j = (date - t_0).days/252
            Z[0].loc[date][col] = np.exp(-r*t_j) * sum(Z[i].loc[date][col] for i in range(1, N+1))/N
            for i in range(1,N+1):
                corrected_path[i].iloc[j][col] = S0[col] * Z[i].iloc[j][col]/Z[0].iloc[j][col]
    return corrected_path[1:]