# cleaning.py
import pandas as pd
import psycopg2
import numpy as np
from sqlalchemy import create_engine, inspect
from sklearn.impute import SimpleImputer
from scipy import stats
import hopsworks
import configparser

class DataCleaning:
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
        self.hopsworks_api_key = self.config.get('API', 'hopswork_api_key')

        self.engine = None
        self.connection = None
        self.dataframes_dict = {}

    def connect_to_postgresql(self):
        """Establish a connection to PostgreSQL and return the engine."""
        try:
            self.engine = create_engine(f'postgresql+psycopg2://{self.db_config["user"]}:{self.db_config["password"]}@{self.db_config["host"]}:{self.db_config["port"]}/{self.db_config["dbname"]}')
            self.connection = self.engine.connect()
            print("Connected to PostgreSQL.")
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            raise

    def read_tables_as_dataframes(self):
        """Fetch all tables from PostgreSQL and load them into a dictionary of DataFrames."""
        if not self.engine:
            self.connect_to_postgresql()

        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()

        for table_name in table_names:
            self.dataframes_dict[table_name] = pd.read_sql_table(table_name, self.connection)
        print("Loaded tables into DataFrames.")

    def remove_null_values(self, df):
        """Handle null values using Simple Imputer (mean for numeric, mode for categorical)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        # Mean imputation for numeric columns
        imputer_numeric = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])

        # Mode imputation for non-numeric columns
        imputer_non_numeric = SimpleImputer(strategy='most_frequent')
        df[non_numeric_cols] = imputer_non_numeric.fit_transform(df[non_numeric_cols])

        return df

    def remove_outliers(self, df, columns, method='IQR'):
        """Remove outliers using IQR or Z-Score method."""
        if method == 'IQR':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method == 'Z-Score':
            z_scores = np.abs(stats.zscore(df[columns]))
            df = df[(z_scores < 3).all(axis=1)]  # Keep only data points within 3 standard deviations
        return df

    def remove_duplicates(self, df):
        """Remove duplicate rows from the DataFrame."""
        return df.drop_duplicates()

    def clean_dataframes(self):
        """Apply the cleaning operations (null values, outliers, duplicates) to all DataFrames."""
        for table_name, df in self.dataframes_dict.items():
            print(f"Cleaning data for table: {table_name}")

            # Step 1: Remove null values
            df = self.remove_null_values(df)

            # Step 2: Remove outliers (for numeric columns only)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = self.remove_outliers(df, columns=numeric_cols)

            # Step 3: Remove duplicates
            df = self.remove_duplicates(df)
            
            # Step 4: Add index_column as a primary key
            df['index_column'] = range(1, len(df) + 1)  # Create a sequential index starting from 1
            
            # Step 5: Set 'event_time' to today's date (don't modify any other columns)
            df['event_time'] = pd.Timestamp.today().normalize()  # This will set the date to today with time set to 00:00:00

            # Step 6: Update the DataFrame in the dataframes_dict after cleaning
            self.dataframes_dict[table_name] = df  # Ensure the cleaned DataFrame is stored back in the dictionary
            print(f"Added 'index_column' and 'event_time' to {table_name}.")

    def connect_to_hopsworks(self):
        """Establish a connection to Hopsworks using the API key."""
        try:
            project = hopsworks.login(api_key_value=self.hopsworks_api_key)
            fs = project.get_feature_store()
            print("Connected to Hopsworks feature store.")
            return fs
        except Exception as e:
            print(f"Error connecting to Hopsworks: {e}")
            raise

    def upload_to_hopsworks(self, fs, version=1):
        """Upload cleaned DataFrames to Hopsworks as feature groups."""
        for table_name, df in self.dataframes_dict.items():
            feature_group_name = f"{table_name}_fg"
            try:
                df.columns = df.columns.str.lower()  # Convert column names to lowercase
                fg = fs.get_or_create_feature_group(
                    name=feature_group_name,
                    version=version,
                    description=f"Feature group for {table_name}",
                    primary_key=['index_column'],
                    event_time='event_time'
                )
                fg.insert(df)
                print(f"Uploaded cleaned data for table '{table_name}' to Hopsworks.")
            except Exception as e:
                print(f"Error uploading data for table '{table_name}': {e}")

    def close_postgresql_connection(self):
        """Close the PostgreSQL connection."""
        if self.connection:
            self.connection.close()
            print("PostgreSQL connection closed.")
