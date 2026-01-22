import numpy as np

def build_features(df):
    df['log_charges'] = np.log1p(df['charges'])
    return df