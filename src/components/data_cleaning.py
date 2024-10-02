import configparser

from constants import CONFIG_FILE_PATH
config = configparser.RawConfigParser()
import sys
import os
import os.path as path
import numpy as np
from sqlalchemy import create_engine, inspect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

# Use raw string to avoid escape character issues
CONFIG_FILE_PATH = r'C:\Desktop\Truck Project\src\config\config.ini'

class DataClean:
    def __init__(self):  # Corrected method name
        self.config = configparser.RawConfigParser()
        self.config.read(CONFIG_FILE_PATH)

        # Extract configuration variables (example; adjust according to your config file)
        self.username = self.config.get('Database', 'username')
        self.password = self.config.get('Database', 'password')
        self.host = self.config.get('Database', 'host')
        self.port = self.config.get('Database', 'port')
        self.database = self.config.get('Database', 'database')

    def read_tables(self):
        # Create SQLAlchemy engine correctly
        engine = create_engine(f'postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}')

        # Use SQLAlchemy's inspector to get all table names
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Dictionary to store dataframes
        dataframes = {}

        # Loop through each table and store it in the dictionary as a dataframe
        for table_name in tables:
            print(f"Reading data from table {table_name}...")

            # Download data from PostgreSQL table
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, engine)

            # Store the dataframe in the dictionary
            dataframes[table_name] = df
            print(f"Data from {table_name} loaded into dataframe.")
        engine.dispose()
        # Return the dictionary containing all dataframes
        return dataframes     

    def detect_and_remove_outliers(self, df, columns, method='IQR'):
        df_cleaned = df.copy()

        if method == 'IQR':
            for column in columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    # Remove outliers
                    df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
                    
        elif method == 'Z-Score':
            for column in columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    mean = df[column].mean()
                    std_dev = df[column].std()
                    lower_bound = mean - 3 * std_dev
                    upper_bound = mean + 3 * std_dev
                    # Remove outliers
                    df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
                    
        else:
            raise ValueError("Unsupported method. Use 'IQR' or 'Z-Score'.")

        return df_cleaned
    
    def filling_missing_values(self, dataframe, n_neighbors=5):
        # Create imputer instances
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)

        # Separate numeric and non-numeric columns
        numeric_cols = dataframe.select_dtypes(include=['number']).columns
        non_numeric_cols = dataframe.select_dtypes(include=['object']).columns
        
        # Process numeric columns
        for col in numeric_cols:
            if dataframe[col].isnull().sum() > 0:
                # Apply KNN imputer to numeric columns with missing values
                dataframe[col] = knn_imputer.fit_transform(dataframe[[col]])
        
        # Process non-numeric columns
        for col in non_numeric_cols:
            if dataframe[col].isnull().sum() > 0:
                # Apply mode imputation
                mode_value = dataframe[col].mode()[0]  # Get the most frequent value
                dataframe[col] = dataframe[col].fillna(mode_value)
        
        return dataframe
    
    def remove_columns(self, df, cols_to_remove):
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns], errors='ignore')
        return df
    
    def transform_and_clean_data(self, df, date_col, hour_col, columns_to_remove):
        # Function to merge date and hour into a single datetime column
        def transform_hour_to_datetime(date, hour):
            hour_str = str(hour).zfill(4)  # Zero-fill the hour to 4 digits (e.g., 0100)
            hour_part = hour_str[:2]       # Extract hour part
            minute_part = hour_str[2:]     # Extract minute part
            time_str = f"{hour_part}:{minute_part}:00"
            return pd.Timestamp(f"{date} {time_str}")
        
        # Apply the transformation
        df['datetime'] = df.apply(lambda row: transform_hour_to_datetime(row[date_col], row[hour_col]), axis=1)
        
        # Remove unwanted columns
        df_cleaned = df.drop(columns=columns_to_remove, axis=1, errors='ignore')  # 'ignore' avoids error if the column doesn't exist
    
        return df_cleaned
    
    def upsert_to_feature_group(self, fs, name, df, primary_key, version=1):
        """
        Upsert data to a Hopsworks feature group, or create the feature group if it does not exist.
        Ensures all columns from the dataframe are included in the feature group schema.
        """
        try:
            # Attempt to get the existing feature group
            fg = fs.get_feature_group(name=name, version=version)
            print(f"Feature group '{name}' exists. Upserting data...")
            fg.insert(df, write_options={"upsert": True})
        except Exception as e:
            # If feature group doesn't exist, create it
            print(f"Feature group '{name}' not found. Creating a new feature group... {e}")

            # Dynamically infer the schema from the dataframe columns
            new_fg = fs.create_feature_group(
                name=name,
                version=version,
                primary_key=primary_key,
                description=f"Feature group for {name}",
                event_time='event_time'
            )

            # Insert all columns from the dataframe into the feature group
            new_fg.insert(df)
            print(f"Feature group '{name}' created and data inserted.")
    
    def reg_catvar(self, df, cols):
        for col in cols:
            df[col] = df[col].apply(lambda x: str(x).lower())
        return df  # Return the modified dataframe

