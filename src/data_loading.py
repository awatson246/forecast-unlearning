import pandas as pd
import os

def load_data(choice):
    if choice == '1':
        return chicago_data_pull()
    elif choice == '2':
        return credit_spending_habits_data_pull()
    elif choice == '3':
        return superstore_sales_data_pull()
    else:
        print("Invalid dataset choice.")
        return None, None

def chicago_data_pull():
    cwd = os.getcwd()
    dataset_path = os.path.join(cwd, 'Datasets', 'chicago_crimes.csv')
    df = pd.read_csv(dataset_path)

    if 'year' in df.columns:
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01') + \
                     pd.to_timedelta(df.groupby('year').cumcount() % 365, unit='D')
        df.drop(columns=['year'], inplace=True)

    columns_to_drop = ['arrest_count', 'false_count', 'description']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    settings = {
        'numeric_columns': ['district', 'latitude', 'longitude', 'crime_count'],
        'categorical_columns': ['primary_type'],
        'datetime_column': 'date',
        'target_column': 'crime_count',
        'location_columns': ['latitude', 'longitude']
        }
    return df, settings

def credit_spending_habits_data_pull():
    cwd = os.getcwd()
    dataset_path = os.path.join(cwd, 'Datasets', 'credit_spending.csv')
    df = pd.read_csv(dataset_path)

    df[['City', 'Country']] = df['City'].str.split(', ', expand=True)
    settings = {
        'numeric_columns': ['Amount'],
        'categorical_columns': ['City', 'Card Type', 'Exp Type', 'Gender', 'Country'],
        'datetime_column': 'Date',
        'target_column': 'Amount',
        'location_columns': ['City', 'Country']
        }
    return df, settings

def superstore_sales_data_pull():
    cwd = os.getcwd()
    dataset_path = os.path.join(cwd, 'Datasets', 'superstore_sales.csv')
    df = pd.read_csv(dataset_path)

    settings = {
        'numeric_columns': ['Postal Code', 'Sales'],
        'categorical_columns': [
            'Order ID', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 
            'Country', 'City', 'State', 'Region', 'Product ID', 'Category', 
            'Sub-Category', 'Product Name'
        ],
        'datetime_column': 'Order Date',
        'target_column': 'Sales',
        'location_columns': ['Postal Code', 'Country', 'City', 'State', 'Region']
        }
    return df, settings
