import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger

logger = get_logger(__name__)

def load_and_process_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with shape: {df.shape}")

        df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)
        df.drop(columns=['Serial_No'], inplace=True)

        df['University_Rating'] = df['University_Rating'].astype('object')
        df['Research'] = df['Research'].astype('object')

        df_encoded = pd.get_dummies(df, drop_first=True)

        X = df_encoded.drop('Admit_Chance', axis=1)
        y = df_encoded['Admit_Chance']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        logger.info("Preprocessing and train-test split completed.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.exception("Data preprocessing failed.")
        raise
