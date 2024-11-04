import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(df, settings):
    """Preprocess the input DataFrame by converting datetime, normalizing numeric columns, 
        and encoding categorical columns."""
    
    # Convert datetime column and extract features
    df[settings['datetime_column']] = pd.to_datetime(
        df[settings['datetime_column']], 
        format='mixed', 
        dayfirst=True, 
        errors='coerce'
    )
    df['year'] = df[settings['datetime_column']].dt.year
    df['month'] = df[settings['datetime_column']].dt.month
    df['day'] = df[settings['datetime_column']].dt.day
    
    # Drop the original datetime column
    df = df.drop(columns=[settings['datetime_column']])

    # Normalize numeric columns
    scaler = MinMaxScaler()
    df[settings['numeric_columns']] = scaler.fit_transform(df[settings['numeric_columns']])
    
    # Drop any rows with NaN values after normalization
    df = df.dropna()

    # Encode categorical columns
    for col in settings['categorical_columns']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df

def create_dataset(df, target_column, look_back=1):
    """Convert a DataFrame into sequences with look_back steps for LSTM input."""
    
    # Ensure all data is numeric and drop the target column for features
    df_numeric = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
    
    if df_numeric.isnull().values.any():
        raise ValueError("Data contains NaN values, which cannot be processed.")

    # Prepare the target array
    target = df[target_column].values  # Only target values for dataY
    
    dataX, dataY = [], []

    # Create sequences of look_back steps
    for i in range(len(df_numeric) - look_back):
        dataX.append(df_numeric.iloc[i:(i + look_back)].values)  # Keep as 2D for LSTM
        dataY.append(target[i + look_back])
    
    # Convert lists to numpy arrays
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    return dataX, dataY
