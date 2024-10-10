import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import hopsworks
import configparser
from datetime import datetime


class DataClean:
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
            print(f"Feature group '{feature_group_name}' already exists. Skipping insertion.")
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
            print(f"Statistics computed for {feature_group.name}.")
