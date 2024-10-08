import os
import hopsworks
import pandas as pd
import sys

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
class DataFetcher:
    def __init__(self):
        # Initialize Hopsworks connection
        self.project = hopsworks.login()
        self.fs = self.project.get_feature_store()

    def fetch_data(self, feature_group_name, version=1):
        """Fetch data from the feature store for a given feature group."""
        feature_group = self.fs.get_feature_group(feature_group_name, version=version)
        query = feature_group.select_all()
        dataframe = query.read()
        return dataframe

    def fetch_all_data(self):
        """Fetch all necessary data from the feature store."""
        dataframes = {
            "city_weather_df": self.fetch_data("city_weather_fg", version=1),
            "route_weather_df": self.fetch_data("routes_weather_fg", version=1),
            "trucks_df": self.fetch_data("trucks_table_fg", version=1),
            "drivers_df": self.fetch_data("drivers_details_fg", version=1),
            "routes_df": self.fetch_data("routes_table_fg", version=1),
            "truck_schedule_df": self.fetch_data("truck_schedule_table_fg", version=1),
            "traffic_df": self.fetch_data("traffic_table_fg", version=1)
        }
        return dataframes


class DataPreparation:
    def __init__(self):
        # Common drop and duplicate configurations
        self.drop_configs = {
            "city_weather_df": {
                "columns_to_drop": ['event_time', 'index_column', 'chanceofrain', 'chanceofsnow', 'chanceofthunder', 'chanceoffog'],
                "duplicate_columns": ['city_id', 'date', 'hour'],
                "has_date_hour": True
            },
            "route_weather_df": {
                "columns_to_drop": ['event_time', 'index_column', 'chanceofrain', 'chanceofsnow', 'chanceofthunder', 'chanceoffog'],
                "duplicate_columns": ['route_id', 'date'],
                "has_date_hour": True
            },
            "trucks_df": {
                "columns_to_drop": ['event_time', 'index_column'],
                "duplicate_columns": ['truck_id'],
                "has_date_hour": False
            },
            "drivers_df": {
                "columns_to_drop": ['event_time', 'index_column'],
                "duplicate_columns": ['driver_id'],
                "has_date_hour": False
            },
            "routes_df": {
                "columns_to_drop": ['event_time', 'index_column'],
                "duplicate_columns": ['route_id', 'destination_id', 'origin_id'],
                "has_date_hour": False
            },
            "truck_schedule_df": {
                "columns_to_drop": ['event_time', 'index_column'],
                "duplicate_columns": ['truck_id', 'route_id', 'departure_date'],
                "has_date_hour": False
            },
            "traffic_df": {
                "columns_to_drop": ['event_time', 'index_column'],
                "duplicate_columns": ['route_id', 'custom_date'],
                "has_date_hour": False
            }
        }
    
    def drop_columns_and_duplicates(self, df, config):
        """Drop unnecessary columns and duplicate rows for a DataFrame."""
        # Drop unnecessary columns
        df.drop(columns=config['columns_to_drop'], errors='ignore', inplace=True)
        # Drop duplicate rows
        df.drop_duplicates(subset=config['duplicate_columns'], inplace=True)
    
    def create_custom_datetime(self, df, date_col='date', hour_col='hour'):
        """Convert date and hour columns to a custom datetime column."""
        df[hour_col] = df[hour_col].apply(lambda x: f"{x:04d}")  # Convert hour to 4-digit string
        df.insert(1, 'custom_date', pd.to_datetime(df[date_col] + ' ' + df[hour_col], format='%Y-%m-%d %H%M'))  # Create custom_date
    
    def prepare_data(self, dataframes):
        """Prepare all DataFrames by applying common cleaning steps."""
        for df_name, df in dataframes.items():
            print(f"Preparing {df_name}...")

            config = self.drop_configs.get(df_name, {})
            if config:
                # Apply column and duplicate handling
                self.drop_columns_and_duplicates(df, config)
                
                # If the dataframe has 'date' and 'hour' columns, create custom datetime
                if config.get('has_date_hour', False):
                    self.create_custom_datetime(df, 'date', 'hour')
            
        return dataframes
