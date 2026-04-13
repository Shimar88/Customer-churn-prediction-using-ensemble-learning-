import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y
