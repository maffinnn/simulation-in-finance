import pandas as pd

def apply_empirical_martingale_correction(df_path):
    corrected_path = pd.DataFrame(index=df_path.index, columns=df_path.columns)
    col = df_path.columns[0]
    corrected_path.iloc[0] = df_path.iloc[0]
    for j in range(1, len(df_path.index)):
        corrected_path.iloc[j][col] = corrected_path.iloc[j-1][col] * df_path.iloc[j][col]/df_path.iloc[j-1][col]
    return corrected_path
