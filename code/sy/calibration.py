import pandas as pd
import numpy as np

def apply_empirical_martingale_correction(df_path_n):
    r = 1.67/100
    N = len(df_path_n)
    df_path = df_path_n[0]
    t_0 = df_path.index[0]
    Z = [pd.DataFrame(index=df_path.index, columns=df_path.columns) for _ in range(N)]
    corrected_path = [pd.DataFrame(index=df_path.index, columns=df_path.columns) for _ in range(N)]
    col = df_path.columns[0]
    for i in range(N):
        corrected_path[i].iloc[0][col] = df_path_n[i].iloc[0][col]
    for j in range(1, len(df_path.index)):
        for i in range(1, N):
            Z[i].iloc[j][col] = corrected_path[i].iloc[j-1][col] * df_path_n[i].iloc[j][col]/df_path_n[i].iloc[j-1][col]
        date = df_path.index[j]
        t_j = (date - t_0).days/252
        Z[0].loc[date][col] = np.exp(-r*t_j) * sum(Z[i].loc[date][col] for i in range(1, N))/N
        for i in range(N):
            corrected_path[i].iloc[j][col] = corrected_path[i].iloc[0][col] * Z[i].iloc[j][col]/Z[0].iloc[j][col]
    return corrected_path
