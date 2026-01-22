import numpy as np
from joblib import load

import os
os.chdir('../..')

from src.utils.paths import MODELS_DIR, PREDICTED_DIR

def full_data(df):
    X = df.drop('log_charges', axis=1)

    loaded_model = load(MODELS_DIR / 'model.joblib')

    all_predictions = loaded_model.predict(X)

    df['charges (predict)'] = np.expm1(all_predictions)

    df.head()
    df.to_csv(PREDICTED_DIR / 'data_predicted.csv')