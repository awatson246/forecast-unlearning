import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, settings):
        self.settings = settings
        self.numeric_scaler = MinMaxScaler()
        self.encoders = {}  # To store label encoders for each categorical column
    
    def preprocess_data(self, df):
        """Preprocess the input DataFrame by converting datetime, normalizing numeric columns, 
           and encoding categorical columns."""
        
        print("Preprocessing data...")
        
        # Convert datetime column and extract features
        df[self.settings['datetime_column']] = pd.to_datetime(
            df[self.settings['datetime_column']], 
            format='mixed', 
            dayfirst=True, 
            errors='coerce'
        )
        df['year'] = df[self.settings['datetime_column']].dt.year
        df['month'] = df[self.settings['datetime_column']].dt.month
        df['day'] = df[self.settings['datetime_column']].dt.day
        
        # Drop the original datetime column
        df = df.drop(columns=[self.settings['datetime_column']])

        # Normalize numeric columns
        df = self.normalize_numeric_columns(df, self.settings['numeric_columns'])
        
        # Encode categorical columns
        df = self.encode_categorical_columns(df, self.settings['categorical_columns'])

        return df

    def normalize_numeric_columns(self, df, numeric_columns):
        """Normalize numeric columns using MinMaxScaler."""
        df[numeric_columns] = self.numeric_scaler.fit_transform(df[numeric_columns])
        return df

    def encode_categorical_columns(self, df, categorical_columns):
        """Encode categorical columns using LabelEncoder."""
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le  # Store the encoder for later use
        return df

    def transform_numeric_columns(self, df):
        """Transform new data using the fitted MinMaxScaler."""
        df[df.columns] = self.numeric_scaler.transform(df[df.columns])
        return df

    def transform_categorical_columns(self, df):
        """Apply stored LabelEncoders to new categorical data."""
        for col, le in self.encoders.items():
            df[col] = le.transform(df[col])
        return df

    def create_dataset(self, df, target_column, look_back=1):
        """Convert a DataFrame into sequences with look_back steps for LSTM input."""
        
        print("Creating dataset...")
        
        # Ensure all data is numeric and drop the target column for features
        df_numeric = df.drop(columns=[target_column]).select_dtypes(include=[np.number])

        # Handle NaN values by filling with the mean of the column (or other strategies)
        df_numeric = df_numeric.fillna(df_numeric.mean())  # Fill NaNs with the column mean

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
