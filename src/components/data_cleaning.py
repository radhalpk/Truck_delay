import configparser
config = configparser.RawConfigParser()
from datetime import datetime
import sys
import os
import os.path as path
import numpy as np
from sqlalchemy import create_engine,inspect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hopsworks
from sklearn.impute import KNNImputer
from src.config import *
import time
from hsfs.client import exceptions as hsfs_exceptions

# Define the path to the configuration file
CONFIG_FILE_PATH = '/Users/pavankumarradhala/Desktop/projects/Truck_delay/src/config/config.ini'

class DataClean:
    def __init__(self):
        # Load config file for database credentials and Hopsworks API key
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_FILE_PATH)

        # PostgreSQL Database Config
        self.db_config = {
            'dbname': self.config.get('Database', 'dbname'),
            'user': self.config.get('Database', 'username'),
            'password': self.config.get('Database', 'password'),
            'host': self.config.get('Database', 'host'),
            'port': self.config.get('Database', 'port')
        }

        # Hopsworks API Key
        self.hopsworks_api_key = self.config.get('API', 'hopswork_api_key')

        # Hopsworks login
        self.project = hopsworks.login(api_key_value=self.hopsworks_api_key)

        # Create SQLAlchemy engine
        self.engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        
    def add_id_and_event_time(self, df, id_col_name='id', event_time_col_name='event_time'):
        """
        Adds 'id' as a sequential index and 'event_time' as the current date to the DataFrame.
        
        Parameters:
        - df: DataFrame to modify
        - id_col_name: Name of the 'id' column to be added (default: 'id')
        - event_time_col_name: Name of the 'event_time' column (default: 'event_time')
        
        Returns:
        - df: Modified DataFrame with 'id' and 'event_time' columns added
        """
        # Add 'id' column if it doesn't exist
        if id_col_name not in df.columns:
            df[id_col_name] = range(1, len(df) + 1)

        # Add 'event_time' column if it doesn't exist, set to current date
        if event_time_col_name not in df.columns:
            df[event_time_col_name] = pd.Timestamp(datetime.today())  # Fix the datetime call

        return df
    
    def read_tables(self):
        """Fetch data from the PostgreSQL database and return all tables as DataFrames."""
        # Use SQLAlchemy's inspector to get all table names
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        # Dictionary to store dataframes
        dataframes = {}

        # Loop through each table and store it in the dictionary as a dataframe
        for table_name in tables:
            print(f"Reading data from table {table_name}...")
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            dataframes[table_name] = df
            print(f"Data from {table_name} loaded into dataframe.")

        # Close the engine connection
        self.engine.dispose()

        # Return the dictionary containing all dataframes
        return dataframes

    def detect_and_remove_outliers(self, df, columns, method='IQR'):
        """
        Detect and remove outliers from the DataFrame using IQR or Z-Score methods.
        """
        df_cleaned = df.copy()

        if method == 'IQR':
            for column in columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
        elif method == 'Z-Score':
            for column in columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    mean = df[column].mean()
                    std_dev = df[column].std()
                    lower_bound = mean - 3 * std_dev
                    upper_bound = mean + 3 * std_dev
                    df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
        else:
            raise ValueError("Unsupported method. Use 'IQR' or 'Z-Score'.")

        return df_cleaned

    def filling_missing_values(self, dataframe, n_neighbors=5):
        """Fill missing values in the DataFrame using KNN for numeric columns and mode for non-numeric."""
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)

        # Separate numeric and non-numeric columns
        numeric_cols = dataframe.select_dtypes(include=['number']).columns
        non_numeric_cols = dataframe.select_dtypes(include=['object']).columns

        # Process numeric columns
        for col in numeric_cols:
            if dataframe[col].isnull().sum() > 0:
                dataframe[col] = knn_imputer.fit_transform(dataframe[[col]])

        # Process non-numeric columns
        for col in non_numeric_cols:
            if dataframe[col].isnull().sum() > 0:
                mode_value = dataframe[col].mode()[0]  # Get the most frequent value
                dataframe[col] = dataframe[col].fillna(mode_value)

        return dataframe

    def remove_columns(self, df, cols_to_remove):
        """Remove unwanted columns from the DataFrame."""
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns], errors='ignore')
        return df

    def transform_and_clean_data(self, df, date_col, hour_col, columns_to_remove):
        """Transform date and hour into a single datetime column and remove unwanted columns."""
        def transform_hour_to_datetime(date, hour):
            hour_str = str(hour).zfill(4)  # Zero-fill the hour to 4 digits (e.g., 0100)
            hour_part = hour_str[:2]       # Extract hour part
            minute_part = hour_str[2:]     # Extract minute part
            time_str = f"{hour_part}:{minute_part}:00"
            return pd.Timestamp(f"{date} {time_str}")

        # Apply the transformation
        df['datetime'] = df.apply(lambda row: transform_hour_to_datetime(row[date_col], row[hour_col]), axis=1)

        # Remove unwanted columns
        df_cleaned = df.drop(columns=columns_to_remove, axis=1, errors='ignore')
    
        return df_cleaned

    '''def upsert_to_feature_group(self, fs, name, df, primary_key, version=1):
        """Upsert data to a Hopsworks feature group or create a new feature group if it doesn't exist."""
        try:
            # Attempt to get the existing feature group
            fg = fs.get_feature_group(name=name, version=version)
            print(f"Feature group '{name}' exists. Upserting data...")
            fg.insert(df, write_options={"upsert": True})
        except:
            # If the feature group doesn't exist, create it
            print(f"Feature group '{name}' not found. Creating a new feature group...")

            # Create the feature group dynamically from the dataframe schema
            new_fg = fs.create_feature_group(
                name=name,
                version=version,
                primary_key=primary_key,
                description=f"Feature group for {name}",
                event_time='event_time'
            )

            # Insert data into the new feature group
            new_fg.insert(df)
            print(f"Feature group '{name}' created and data inserted.")'''
   

    

    def upsert_to_feature_group(self, fs, name, df, primary_key, version=1, max_retries=3, retry_delay=60):
       """Upsert data to a Hopsworks feature group or create a new feature group if it doesn't exist."""
       try:
           # Attempt to get the existing feature group
           fg = fs.get_feature_group(name=name, version=version)
           print(f"Feature group '{name}' exists. Upserting data...")
        
        # Try inserting data and handle parallel execution quota issues with retries
           for attempt in range(max_retries):
               try:
                   fg.insert(df, write_options={"upsert": True})
                   print(f"Data upserted successfully to '{name}'.")
                   break
               except hsfs_exceptions.RestAPIError as e:
                   if 'Parallel executions quota reached' in str(e):
                       print(f"Job execution quota reached, retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                       time.sleep(retry_delay)
                   else:
                       raise e
           else:
               raise RuntimeError(f"Failed to upsert data to '{name}' after {max_retries} attempts.")
    
       except hsfs_exceptions.RestAPIError as e:
        # If the feature group doesn't exist, create it
           if 'Feature group not found' in str(e):
               print(f"Feature group '{name}' not found. Creating a new feature group...")

            # Create the feature group dynamically from the dataframe schema
               new_fg = fs.create_feature_group(
                name=name,
                version=version,
                primary_key=primary_key,
                description=f"Feature group for {name}",
                event_time='event_time'
            )

            # Insert data into the new feature group
               new_fg.insert(df)
               print(f"Feature group '{name}' created and data inserted.")
           else:
            # For any other errors, re-raise the exception
               raise e
       except Exception as e:
           print(f"Error upserting data for {name}: {e}")
           raise e


    def reg_catvar(self, df, cols):
        """Regularize categorical variables by converting them to lowercase strings."""
        for col in cols:
            df[col] = df[col].apply(lambda x: str(x).lower())
        return df



'''class DataClean:
    def __init__(self, config_path):
        # Load config file for database credentials and Hopsworks API key
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # PostgreSQL Database Config
        self.db_config = {
            'dbname': self.config.get('Database', 'dbname'),
            'user': self.config.get('Database', 'username'),
            'password': self.config.get('Database', 'password'),
            'host': self.config.get('Database', 'host'),
            'port': self.config.get('Database', 'port')
        }

        # Hopsworks API Key
        self.project = hopsworks.login(
            api_key_value=self.config['API']['hopswork_api_key']
        )

        # Create the database connection
        self.engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        self.feature_store = self.project.get_feature_store()

    def read_table(self, table_name):
        """Fetch data from the PostgreSQL database."""
        return pd.read_sql_table(table_name, self.engine)

    def drop_duplicates(self, df):
        """Remove duplicate rows."""
        return df.drop_duplicates()

    def remove_outliers_iqr(self, df):
        """Remove outliers using IQR method for numeric columns."""
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    def add_index_column(self, df):
        """Adds an index column."""
        df.reset_index(drop=False, inplace=True)
        df['index'] = df['index'].astype('int32')  # Ensure index is int32
        return df

    def add_event_date(self, df):
        """Adds an event date column."""
        df['event_date'] = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
        return df

    def merge_date_hour(self, df):
        """Merges date and hour columns into a single datetime column, then removes the original date and hour columns."""
        # Check if the column is 'Date' and convert it to lowercase for compatibility
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        if 'hour' in df.columns:
            df['hour'] = df['hour'].apply(lambda x: f"{int(x):04d}" if pd.notnull(x) else '0000')
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'], errors='coerce')

        # Remove the original 'date' and 'hour' columns
        df.drop(columns=['date', 'hour'], inplace=True, errors='ignore')

        return df

    def sanitize_dataframe(self, df):
        """Sanitize DataFrame columns by converting them to lowercase and removing special characters."""
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
        return df

    def save_cleaned_data_hopsworks(self, df, table_name):
        """Save the cleaned data to Hopsworks feature store."""
        feature_group_mapping = {
            'city_weather': 'city_weather_features',
            'drivers_table': 'drivers_features',
            'routes_table': 'routes_features',
            'routes_weather': 'routes_weather_features',
            'traffic_table': 'traffic_table_features',
            'trucks_table': 'trucks_features',
            'truck_schedule_table': 'truck_schedule_features'
        }
        
        feature_group_name = feature_group_mapping.get(table_name, f"{table_name}_features")
        
        # Ensure correct data types
        df['index'] = df['index'].astype('int32')
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

        # Convert 'float64' columns to 'float32'
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        # Convert 'int64' columns to 'int32'
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        try:
            feature_group = self.feature_store.get_feature_group(feature_group_name, version=1)
            feature_group.insert(df, write_options={"upsert": True})
        except:
            print(f"Creating new feature group '{feature_group_name}'.")
            try:
                feature_group = self.feature_store.create_feature_group(
                    name=feature_group_name,
                    version=1,
                    description=f"Cleaned data for {table_name}",
                    primary_key=['index'],
                    event_time='event_date'
                )
                feature_group.insert(df, write_options={"wait_for_job": False})
                print(f"Data saved to new feature group '{feature_group_name}'.")
            except Exception as e:
                print(f"Error saving data to Hopsworks: {e}")

    def configure_and_compute_statistics(self, feature_group):
        """Compute statistics for feature groups."""
        commits = feature_group.get_commits()
        if commits:
            feature_group.compute_statistics()
            print(f"Statistics computed for {feature_group.name}.")'''
