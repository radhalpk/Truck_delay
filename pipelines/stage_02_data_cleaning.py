'''import os
import sys
import pandas as pd
from sqlalchemy import inspect
import hopsworks

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.components.data_cleaning import DataClean

STAGE_NAME = "Data Cleaning"
cleaning_obj = DataClean()

# Initialize Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

class DataCleaningPipeline:
    def __init__(self):
        # Define constants here for reuse
        self.outlier_columns_dict = {
            'city_weather': ['temp', 'wind_speed', 'precip', 'humidity', 'visibility', 'pressure'],
            'drivers_table': ['vehicle_no'],
            'routes_table': ['distance', 'average_hours'],
            'trucks_table': ['truck_id'],
            'truck_schedule_table': ['truck_id'],
            'routes_weather': ['temp', 'wind_speed', 'humidity', 'pressure']
        }
        self.columns_to_remove_dict = {
            'city_weather': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder'],
            'routes_weather': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder']
        }
        # Generalize primary keys for each dataframe
        self.primary_keys = {
            'city_weather': ['id'],
            'drivers_table': ['id'],
            'routes_table': ['id'],
            'trucks_table': ['id'],
            'truck_schedule_table': ['id'],
            'traffic_table': ['id'],
            'routes_weather': ['id']
        }

    def process_dataframe(self, name, df):
        # Step 1: Fill missing values
        cleaned_df = cleaning_obj.filling_missing_values(df)
        
        # Step 2: Handle outliers if applicable
        outlier_cols = self.outlier_columns_dict.get(name, [])
        if outlier_cols:
            cleaned_df = cleaning_obj.detect_and_remove_outliers(cleaned_df, outlier_cols)
        
        # Step 3: Remove unwanted columns if applicable
        cols_to_remove = self.columns_to_remove_dict.get(name, [])
        if cols_to_remove:
            cleaned_df = cleaning_obj.remove_columns(cleaned_df, cols_to_remove)
        
        return cleaned_df

    def transform_and_clean_special_cases(self, dataframes):
        # Special transformations for 'city_weather' and 'traffic_table'
        dataframes['city_weather'] = cleaning_obj.transform_and_clean_data(
            dataframes['city_weather'], 'date', 'hour', ['date', 'hour'])
        dataframes['traffic_table'] = cleaning_obj.transform_and_clean_data(
            dataframes['traffic_table'], 'date', 'hour', ['date', 'hour'])

        # Processing 'truck_schedule_table' for datetime conversions
        truck_schedule_df = dataframes['truck_schedule_table']
        truck_schedule_df['departure_date'] = pd.to_datetime(truck_schedule_df['departure_date'])
        truck_schedule_df['estimated_arrival'] = pd.to_datetime(truck_schedule_df['estimated_arrival']).dt.floor('s')
        dataframes['truck_schedule_table'] = truck_schedule_df

        # Ensure positive experience values in 'drivers_table'
        drivers_df = dataframes['drivers_table']
        drivers_df['experience'] = drivers_df['experience'].abs()
        dataframes['drivers_table'] = drivers_df
        
        route_weather_df=dataframes['routes_weather']
        route_weather_df['Date']=pd.to_datetime(route_weather_df['Date'])
        dataframes['routes_weather']=route_weather_df 
        
    def upsert_feature_groups(self, dataframes):
        # Loop over all dataframes and upsert to feature store
        for name, df in dataframes.items():
            print(f"Upserting data for {name}")
            primary_key = self.primary_keys.get(name, ['id'])  # Default to 'id' if not provided
            cleaning_obj.upsert_to_feature_group(fs, name, df, primary_key)

    def add_id_and_event_time_to_all(self, dataframes):
        """
        Add 'id' and 'event_time' columns to all dataframes.
        
        Parameters:
        - dataframes: Dictionary of DataFrames to process
        """
        for name, df in dataframes.items():
            print(f"Adding 'id' and 'event_time' to {name}...")
            dataframes[name] = cleaning_obj.add_id_and_event_time(df)
            
    def main(self):
        try:
            # Read the initial dataset
            dataframes = cleaning_obj.read_tables()
            self.add_id_and_event_time_to_all(dataframes)
            # Process all dataframes in a single loop
            for name, df in dataframes.items():
                print(f"Processing {name}")
                cleaned_df = self.process_dataframe(name, df)
                dataframes[name] = cleaned_df

            # Handle special transformations
            self.transform_and_clean_special_cases(dataframes)

            # Upsert processed data to feature groups
            self.upsert_feature_groups(dataframes)

        except Exception as e:
            print(f"Error in Data Cleaning Pipeline: {e}")
            raise e

if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :", STAGE_NAME)
        obj = DataCleaningPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e
'''
import os
import sys
import pandas as pd
from sqlalchemy import inspect
import hopsworks

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.components.data_cleaning import DataClean
CONFIG_FILE_PATH = os.path.join(project_root, 'src', 'config', 'config.ini')

STAGE_NAME = "Data Cleaning"
cleaning_obj = DataClean()

# Initialize Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

class DataCleaningPipeline:
    def __init__(self):
        self.outlier_columns_dict = {
            'city_weather': ['temp', 'wind_speed', 'precip', 'humidity', 'visibility', 'pressure'],
            'drivers_table': ['vehicle_no'],
            'routes_table': ['distance', 'average_hours'],
            'trucks_table': ['truck_id'],
            'truck_schedule_table': ['truck_id'],
            'routes_weather': ['temp', 'wind_speed', 'humidity', 'pressure']
        }
        self.columns_to_remove_dict = {
            'city_weather': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder'],
            'routes_weather': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder']
        }
        self.primary_keys = {
            'city_weather': ['id'],
            'drivers_table': ['id'],
            'routes_table': ['id'],
            'trucks_table': ['id'],
            'truck_schedule_table': ['id'],
            'traffic_table': ['id'],
            'routes_weather': ['id']
        }

    def process_dataframe(self, name, df):
        cleaned_df = cleaning_obj.filling_missing_values(df)
        outlier_cols = self.outlier_columns_dict.get(name, [])
        if outlier_cols:
            cleaned_df = cleaning_obj.detect_and_remove_outliers(cleaned_df, outlier_cols)
        cols_to_remove = self.columns_to_remove_dict.get(name, [])
        if cols_to_remove:
            cleaned_df = cleaning_obj.remove_columns(cleaned_df, cols_to_remove)
        return cleaned_df

    def transform_and_clean_special_cases(self, dataframes):
        dataframes['city_weather'] = cleaning_obj.transform_and_clean_data(
            dataframes['city_weather'], 'date', 'hour', ['date', 'hour'])
        dataframes['traffic_table'] = cleaning_obj.transform_and_clean_data(
            dataframes['traffic_table'], 'date', 'hour', ['date', 'hour'])

        truck_schedule_df = dataframes['truck_schedule_table']
        truck_schedule_df['departure_date'] = pd.to_datetime(truck_schedule_df['departure_date'])
        truck_schedule_df['estimated_arrival'] = pd.to_datetime(truck_schedule_df['estimated_arrival']).dt.floor('s')
        dataframes['truck_schedule_table'] = truck_schedule_df

        drivers_df = dataframes['drivers_table']
        drivers_df['experience'] = drivers_df['experience'].abs()
        dataframes['drivers_table'] = drivers_df

        route_weather_df = dataframes['routes_weather']
        route_weather_df['Date'] = pd.to_datetime(route_weather_df['Date'])
        dataframes['routes_weather'] = route_weather_df

    def upsert_feature_groups(self, dataframes):
        for name, df in dataframes.items():
            print(f"Upserting data for {name}")
            primary_key = self.primary_keys.get(name, ['id'])
            cleaning_obj.upsert_to_feature_group(fs, name, df, primary_key)

    def add_id_and_event_time_to_all(self, dataframes):
        for name, df in dataframes.items():
            print(f"Adding 'id' and 'event_time' to {name}...")
            dataframes[name] = cleaning_obj.add_id_and_event_time(df)

    def main(self):
        try:
            dataframes = cleaning_obj.read_tables()
            self.add_id_and_event_time_to_all(dataframes)
            for name, df in dataframes.items():
                cleaned_df = self.process_dataframe(name, df)
                dataframes[name] = cleaned_df
            self.transform_and_clean_special_cases(dataframes)
            self.upsert_feature_groups(dataframes)
        except Exception as e:
            print(f"Error in Data Cleaning Pipeline: {e}")
            raise e

if __name__ == '__main__':
    try:
        print(f">>>>>> Stage started <<<<<< : {STAGE_NAME}")
        obj = DataCleaningPipeline()
        obj.main()
        print(f">>>>>> Stage completed <<<<<< {STAGE_NAME}")
    except Exception as e:
        print(e)
        raise e
