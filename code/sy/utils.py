import json
import pandas as pd

def convert_from_json_to_df(filepath: str)->pd.DataFrame:
    with open(filepath, 'r') as f:
        d = json.load(f)
        df = pd.DataFrame.from_dict(d)
    return df
