import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(df, settings):
    # Convert datetime column and extract features
    df[settings['datetime_column']] = pd.to_datetime(df[settings['datetime_column']])
    df['year'] = df[settings['datetime_column']].dt.year
    df['month'] = df[settings['datetime_column']].dt.month
    df['day'] = df[settings['datetime_column']].dt.day
    df = df.drop(settings['datetime_column'], axis=1)

    # Normalize numeric columns
    scaler = MinMaxScaler()
    df[settings['numeric_columns']] = scaler.fit_transform(df[settings['numeric_columns']])

    # Encode categorical columns
    for col in settings['categorical_columns']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df

def create_dataset(df, target_column, look_back=1):
    data = df.values
    target = df[target_column].values
    dataX, dataY = [], []

    for i in range(len(data) - look_back):
        dataX.append(data[i:(i + look_back)].flatten())  # Flatten the 3D data into 1D
        dataY.append(target[i + look_back])
    
    # Convert lists to numpy arrays
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # Create a DataFrame for X with the appropriate number of features
    num_features = data.shape[1]  # Total number of features
    columns = [f'feature_{i}' for i in range(look_back * num_features)]  # Update columns to reflect the flattened features
    df_X = pd.DataFrame(dataX, columns=columns)

    # Create a DataFrame for Y
    df_Y = pd.DataFrame(dataY, columns=[target_column])

    # Concatenate X and Y DataFrames
    df_result = pd.concat([df_X, df_Y], axis=1)

    return df_result