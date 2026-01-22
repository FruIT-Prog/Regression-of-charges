from src.data.full_data import full_data
from src.utils.paths import RAW_DATA_DIR
from src.models.train import train

if __name__ == '__main__':
    path = RAW_DATA_DIR / 'data.csv'
    df_builded = train(path)
    full_data(df_builded)