from datetime import datetime
import pandas as pd
import numpy as np
import time
import json
import configparser
import hsfs
from bs4 import BeautifulSoup  # Missing import added

# Define the path to the configuration file
CONFIG_FILE_PATH = r'C:\Desktop\Truck Project\src\config\config.ini'
config = configparser.RawConfigParser()
config.read(CONFIG_FILE_PATH)

class FeatureEngineering:
    def __init__(self):
        self.config = config

    def read_data_from_hopsworks(self, feature_store, feature_group_names):
        try:
            feature_group_dataframes = {}
            for name in feature_group_names:
                try:
                    fg_metadata = feature_store.get_feature_group(name, version=1)
                    fg_df = fg_metadata.read() if not isinstance(fg_metadata, pd.DataFrame) else fg_metadata
                    feature_group_dataframes[name] = fg_df
                except Exception as e:
                    print(f"Error reading feature group {name}: {e}")
            return feature_group_dataframes
        except Exception as e:
            print(f"Error retrieving feature groups: {e}")
            return

    def remove_columns(self, dataframes, columns_to_remove):
        for df_name, df in dataframes.items():
            df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
            print(f"Removed columns {columns_to_remove} from {df_name}. Remaining columns: {df.columns.tolist()}")
        return dataframes

    def remove_duplicates(self, duplicate_columns, dataframes):
        for df_name, columns in duplicate_columns.items():
            if df_name in dataframes:
                dataframes[df_name] = dataframes[df_name].drop_duplicates(subset=columns)
                print(f"Dropped duplicates from {df_name} based on {columns}")
        return dataframes

    def custom_mode(self, x):
        return x.mode().iloc[0] if not x.mode().empty else np.nan

    def getting_schedule_weather_data(self, schedule_df_cleaned, route_weather_df_cleaned):
        schedule_df_cleaned['estimated_arrival'] = schedule_df_cleaned['estimated_arrival'].dt.ceil('6H')
        schedule_df_cleaned['departure_date'] = schedule_df_cleaned['departure_date'].dt.floor('6H')
        schedule_df_cleaned['date'] = [pd.date_range(start=row['departure_date'], end=row['estimated_arrival'], freq='6H').tolist() for _, row in schedule_df_cleaned.iterrows()]
        expanded_schedule_df = schedule_df_cleaned.explode('date').reset_index(drop=True)
        schedule_weather_df = pd.merge(expanded_schedule_df, route_weather_df_cleaned, on=['route_id', 'date'], how='left', validate="many_to_many")
        schedule_weather_df = schedule_weather_df.drop_duplicates()

        try:
            schedule_weather_grp = schedule_weather_df.groupby(['truck_id', 'route_id'], as_index=False).agg(
                route_avg_temp=('temp', 'mean'),
                route_avg_wind_speed=('wind_speed', 'mean'),
                route_avg_precip=('precip', 'mean'),
                route_avg_humidity=('humidity', 'mean'),
                route_avg_visibility=('visibility', 'mean'),
                route_avg_pressure=('pressure', 'mean'),
                route_description=('description', self.custom_mode)
            )
        except Exception as e:
            print(f"Error during groupby: {e}")
            schedule_weather_grp = pd.DataFrame()

        schedule_weather_merge = pd.merge(expanded_schedule_df, schedule_weather_grp, on=['truck_id', 'route_id'], how='left', validate="many_to_many")
        schedule_weather_merge = schedule_weather_merge.drop_duplicates().ffill()
        return schedule_weather_merge

    def nearest_hour_schedule_route(self, truck_schedule_df, route_df_cleaned):
        nearest_hour_schedule_df = truck_schedule_df.copy()

        nearest_hour_schedule_df['estimated_arrival_nearest_hour'] = nearest_hour_schedule_df['estimated_arrival'].dt.round("H")
        nearest_hour_schedule_df['departure_date_nearest_hour'] = nearest_hour_schedule_df['departure_date'].dt.round("H")

        nearest_hour_schedule_route_df = pd.merge(nearest_hour_schedule_df, route_df_cleaned, on='route_id', how='left', validate="many_to_many")

        for col in nearest_hour_schedule_route_df.columns:
            if isinstance(nearest_hour_schedule_route_df[col].iloc[0], list):
                nearest_hour_schedule_route_df[col] = nearest_hour_schedule_route_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

        nearest_hour_schedule_route_df = nearest_hour_schedule_route_df.ffill().drop_duplicates()
        return nearest_hour_schedule_route_df, nearest_hour_schedule_df

    def origin_weather_data(self, weather_df_cleaned, nearest_hour_schedule_route_df):
        origin_weather_data = weather_df_cleaned.copy()
        origin_weather_merge = pd.merge(nearest_hour_schedule_route_df, origin_weather_data, left_on=['origin_id', 'departure_date_nearest_hour'], right_on=['city_id', 'datetime'], how='left', validate="many_to_many")
        origin_weather_merge = origin_weather_merge.drop(columns='datetime').ffill().drop_duplicates()
        return origin_weather_merge

    def origin_destination_weather(self, origin_weather_merge_clean, weather_df_cleaned):
        destination_weather_data = weather_df_cleaned.copy()
        origin_destination_weather_merge = pd.merge(origin_weather_merge_clean, destination_weather_data, left_on=['destination_id', 'estimated_arrival_nearest_hour'], right_on=['city_id', 'datetime'], how='left', suffixes=('_origin', '_destination'), validate="many_to_many")
        origin_destination_weather_merge = origin_destination_weather_merge.drop(columns=['datetime', 'city_id_origin', 'city_id_destination']).drop_duplicates().ffill()
        return origin_destination_weather_merge

    def scheduled_route_traffic(self, nearest_hour_schedule_df, traffic_df_clean):
        hourly_exploded_scheduled_df = (
            nearest_hour_schedule_df.assign(custom_date=[pd.date_range(start, end, freq='H') for start, end in zip(nearest_hour_schedule_df['departure_date'], nearest_hour_schedule_df['estimated_arrival'])])
            .explode('custom_date', ignore_index=True)
        )

        scheduled_traffic = hourly_exploded_scheduled_df.merge(traffic_df_clean, left_on=['route_id', 'custom_date'], right_on=['route_id', 'datetime'], how='left')

        def custom_agg(values):
            return int(any(values == 1))

        scheduled_route_traffic = scheduled_traffic.groupby(['truck_id', 'route_id'], as_index=False).agg(
            avg_no_of_vehicles=('no_of_vehicles', 'mean'),
            accident=('accident', custom_agg)
        ).drop_duplicates()
        return scheduled_route_traffic

    def final_merge(self, origin_destination_weather_df, scheduled_route_traffic, schedule_weather_merge, trucks_df_cleaned, drivers_df_cleaned):
        origin_destination_weather_traffic_merge = origin_destination_weather_df.merge(scheduled_route_traffic, on=['truck_id', 'route_id'], how='left').drop_duplicates()
        merged_data_weather_traffic = pd.merge(schedule_weather_merge, origin_destination_weather_traffic_merge, on=['truck_id', 'route_id', 'delay', 'departure_date', 'estimated_arrival'], how='left', validate="many_to_many")
        merged_data_weather_traffic_trucks = pd.merge(merged_data_weather_traffic, trucks_df_cleaned, on='truck_id', how='left', validate="many_to_many")
        final_merge = pd.merge(merged_data_weather_traffic_trucks, drivers_df_cleaned, left_on='truck_id', right_on='vehicle_no', how='left', validate="many_to_many")

        def has_midnight(start, end):
            return int(start.date() != end.date())

        final_merge['is_midnight'] = final_merge.apply(lambda row: has_midnight(row['departure_date'], row['estimated_arrival']), axis=1)
        final_merge = final_merge.drop(columns=['date_x', 'date_y']).drop_duplicates().dropna()

        if 'unique_id' not in final_merge.columns:
            final_merge['unique_id'] = range(1, len(final_merge) + 1)

        final_merge = self.convert_column_types(final_merge)
        return final_merge

    def convert_column_types(self, final_df):
        datetime_columns = [
            'departure_date', 
            'estimated_arrival', 
            'estimated_arrival_nearest_hour', 
            'departure_date_nearest_hour'
        ]

        for col in datetime_columns:
            if pd.api.types.is_datetime64tz_dtype(final_df[col]):
                final_df[col] = final_df[col].dt.tz_localize(None)

        final_df = final_df.astype({
            'unique_id': 'int64',
            'truck_id': 'int64',
            'route_id': 'object',
            'departure_date': 'datetime64[ns]',
            'estimated_arrival': 'datetime64[ns]',
            'delay': 'int64',
            'route_avg_temp': 'float64',
            'route_avg_wind_speed': 'float64',
            'route_avg_precip': 'float64',
            'route_avg_humidity': 'float64',
            'route_avg_visibility': 'float64',
            'route_avg_pressure': 'float64',
            'route_description': 'object',
            'estimated_arrival_nearest_hour': 'datetime64[ns]',
            'departure_date_nearest_hour': 'datetime64[ns]',
            'origin_id': 'object',
            'destination_id': 'object',
            'distance': 'float64',
            'average_hours': 'float64',
            'temp_origin': 'float64',
            'wind_speed_origin': 'float64',
            'description_origin': 'object',
            'precip_origin': 'float64',
            'humidity_origin': 'float64',
            'visibility_origin': 'float64',
            'pressure_origin': 'float64',
            'temp_destination': 'float64',
            'wind_speed_destination': 'float64',
            'description_destination': 'object',
            'precip_destination': 'float64',
            'humidity_destination': 'float64',
            'visibility_destination': 'float64',
            'pressure_destination': 'float64',
            'avg_no_of_vehicles': 'float64',
            'accident': 'int64',
            'truck_age': 'int64',
            'load_capacity_pounds': 'float64',
            'mileage_mpg': 'int64',
            'fuel_type': 'object',
            'driver_id': 'object',
            'name': 'object',
            'gender': 'object',
            'age': 'int64',
            'experience': 'int64',
            'driving_style': 'object',
            'ratings': 'int64',
            'vehicle_no': 'int64',
            'average_speed_mph': 'float64',
            'is_midnight': 'int64'
        })
        return final_df

    def upsert_finaldf(self, fs, final_df):
        try:
            final_df['unique_id'] = final_df['unique_id'].astype('int32')
            final_df['accident'] = final_df['accident'].astype('float64')

            print("Final DataFrame before upsert:")
            print(final_df)

            fg = fs.get_feature_group(name="final_df_feature_group", version=1)
            print("Feature group exists. Upserting data...")
            
            try:
                fg.insert(final_df, write_options={"upsert": True})
            except hsfs.client.exceptions.RestAPIError as e:
                if "Parallel executions quota reached" in str(e):
                    print("Max parallel executions reached. Retrying after a pause...")
                    time.sleep(60)
                    fg.insert(final_df, write_options={"upsert": True})
                else:
                    print(f"Error inserting data into feature group: {e}")
        except hsfs.client.exceptions.FeatureStoreException as e:
            print(f"Error retrieving feature group: {e}")
            if 'Feature group not found' in str(e):
                print("Feature group not found. Creating a new feature group...")
                new_fg = fs.create_feature_group(
                    name="final_df_feature_group",
                    version=1,
                    primary_key=['unique_id'],
                    description="Feature group for Final Data",
                )
                new_fg.insert(final_df)
                print("Feature group created and data inserted.")
            else:
                print(f"Error in feature group operations: {e}")





'''from datetime import datetime
import pandas as pd
import numpy as np
import time
import json
import configparser
import hsfs
from bs4 import BeautifulSoup  # Missing import added

# Define the path to the configuration file
CONFIG_FILE_PATH = r'C:\Desktop\Truck Project\src\config\config.ini'
config = configparser.RawConfigParser()
config.read(CONFIG_FILE_PATH)

class FeatureEngineering:
    def __init__(self):
        self.config = config

    def read_data_from_hopsworks(self, feature_store, feature_group_names):
        try:
            feature_group_dataframes = {}
            for name in feature_group_names:
                try:
                    fg_metadata = feature_store.get_feature_group(name, version=1)
                    fg_df = fg_metadata.read() if not isinstance(fg_metadata, pd.DataFrame) else fg_metadata
                    feature_group_dataframes[name] = fg_df
                except Exception as e:
                    print(f"Error reading feature group {name}: {e}")
            return feature_group_dataframes
        except Exception as e:
            print(f"Error retrieving feature groups: {e}")
            return 

    def remove_columns(self, dataframes, columns_to_remove):
        for df_name, df in dataframes.items():
            df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
            print(f"Removed columns {columns_to_remove} from {df_name}. Remaining columns: {df.columns.tolist()}")
        return dataframes

    def remove_duplicates(self, duplicate_columns, dataframes):
        for df_name, columns in duplicate_columns.items():
            if df_name in dataframes:
                dataframes[df_name] = dataframes[df_name].drop_duplicates(subset=columns)
                print(f"Dropped duplicates from {df_name} based on {columns}")
        return dataframes

    def custom_mode(self, x):
        return x.mode().iloc[0] if not x.mode().empty else np.nan

    def getting_schedule_weather_data(self, schedule_df_cleaned, route_weather_df_cleaned):
        schedule_df_cleaned['estimated_arrival'] = schedule_df_cleaned['estimated_arrival'].dt.ceil('6H')
        schedule_df_cleaned['departure_date'] = schedule_df_cleaned['departure_date'].dt.floor('6H')
        schedule_df_cleaned['date'] = [pd.date_range(start=row['departure_date'], end=row['estimated_arrival'], freq='6H').tolist() for _, row in schedule_df_cleaned.iterrows()]
        expanded_schedule_df = schedule_df_cleaned.explode('date').reset_index(drop=True)
        schedule_weather_df = pd.merge(expanded_schedule_df, route_weather_df_cleaned, on=['route_id', 'date'], how='left', validate="many_to_many")
        schedule_weather_df = schedule_weather_df.drop_duplicates()

        try:
            schedule_weather_grp = schedule_weather_df.groupby(['truck_id', 'route_id'], as_index=False).agg(
                route_avg_temp=('temp', 'mean'),
                route_avg_wind_speed=('wind_speed', 'mean'),
                route_avg_precip=('precip', 'mean'),
                route_avg_humidity=('humidity', 'mean'),
                route_avg_visibility=('visibility', 'mean'),
                route_avg_pressure=('pressure', 'mean'),
                route_description=('description', self.custom_mode)
            )
        except Exception as e:
            print(f"Error during groupby: {e}")
            schedule_weather_grp = pd.DataFrame()

        schedule_weather_merge = pd.merge(expanded_schedule_df, schedule_weather_grp, on=['truck_id', 'route_id'], how='left', validate="many_to_many")
        schedule_weather_merge = schedule_weather_merge.drop_duplicates().ffill()
        return schedule_weather_merge

    def nearest_hour_schedule_route(self, truck_schedule_df, route_df_cleaned):
       nearest_hour_schedule_df = truck_schedule_df.copy()
    
    # Round to nearest hour
       nearest_hour_schedule_df['estimated_arrival_nearest_hour'] = nearest_hour_schedule_df['estimated_arrival'].dt.round("H")
       nearest_hour_schedule_df['departure_date_nearest_hour'] = nearest_hour_schedule_df['departure_date'].dt.round("H")
    
    # Merge with route_df_cleaned
       nearest_hour_schedule_route_df = pd.merge(nearest_hour_schedule_df, route_df_cleaned, on='route_id', how='left', validate="many_to_many")
    
    # Convert list-like columns to strings before dropping duplicates
       for col in nearest_hour_schedule_route_df.columns:
        if isinstance(nearest_hour_schedule_route_df[col].iloc[0], list):
            nearest_hour_schedule_route_df[col] = nearest_hour_schedule_route_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
    
    # Now drop duplicates and handle missing values
       #nearest_hour_schedule_route_df = nearest_hour_schedule_route_df.dropna().drop_duplicates()
       #return nearest_hour_schedule_route_df, nearest_hour_schedule_df
   
    # Now drop duplicates and handle missing values
       nearest_hour_schedule_route_df = nearest_hour_schedule_route_df.ffill().drop_duplicates()
       return nearest_hour_schedule_route_df, nearest_hour_schedule_df

    def origin_weather_data(self, weather_df_cleaned, nearest_hour_schedule_route_df):
        origin_weather_data = weather_df_cleaned.copy()
        origin_weather_merge = pd.merge(nearest_hour_schedule_route_df, origin_weather_data, left_on=['origin_id','departure_date_nearest_hour'], right_on=['city_id','datetime'], how='left', validate="many_to_many")
        #origin_weather_merge = origin_weather_merge.drop(columns='datetime').dropna().drop_duplicates()
        #return origin_weather_merge
        origin_weather_merge = origin_weather_merge.drop(columns='datetime').ffill().drop_duplicates()
        return origin_weather_merge

    def origin_destination_weather(self, origin_weather_merge_clean, weather_df_cleaned):
        destination_weather_data = weather_df_cleaned.copy()
        origin_destination_weather_merge = pd.merge(origin_weather_merge_clean, destination_weather_data, left_on=['destination_id','estimated_arrival_nearest_hour'], right_on=['city_id','datetime'], how='left', suffixes=('_origin', '_destination'), validate="many_to_many")
        origin_destination_weather_merge = origin_destination_weather_merge.drop(columns=['datetime','city_id_origin','city_id_destination']).drop_duplicates().ffill()
        return origin_destination_weather_merge

    def scheduled_route_traffic(self, nearest_hour_schedule_df, traffic_df_clean):
        hourly_exploded_scheduled_df = (
            nearest_hour_schedule_df.assign(custom_date=[pd.date_range(start, end, freq='H') for start, end in zip(nearest_hour_schedule_df['departure_date'], nearest_hour_schedule_df['estimated_arrival'])])
            .explode('custom_date', ignore_index=True)
        )
        scheduled_traffic = hourly_exploded_scheduled_df.merge(traffic_df_clean, left_on=['route_id','custom_date'], right_on=['route_id','datetime'], how='left')

        def custom_agg(values):
            return int(any(values == 1))

        scheduled_route_traffic = scheduled_traffic.groupby(['truck_id', 'route_id'], as_index=False).agg(
            avg_no_of_vehicles=('no_of_vehicles', 'mean'),
            accident=('accident', custom_agg)
        ).drop_duplicates()
        return scheduled_route_traffic

    def final_merge(self, origin_destination_weather_df, scheduled_route_traffic, schedule_weather_merge, trucks_df_cleaned, drivers_df_cleaned):
        origin_destination_weather_traffic_merge = origin_destination_weather_df.merge(scheduled_route_traffic, on=['truck_id', 'route_id'], how='left').drop_duplicates()
        merged_data_weather_traffic = pd.merge(schedule_weather_merge, origin_destination_weather_traffic_merge, on=['truck_id', 'route_id','delay','departure_date','estimated_arrival'], how='left', validate="many_to_many")
        merged_data_weather_traffic_trucks = pd.merge(merged_data_weather_traffic, trucks_df_cleaned, on='truck_id', how='left', validate="many_to_many")
        final_merge = pd.merge(merged_data_weather_traffic_trucks, drivers_df_cleaned, left_on='truck_id', right_on='vehicle_no', how='left', validate="many_to_many")
        
        def has_midnight(start, end):
            return int(start.date() != end.date())

        final_merge['is_midnight'] = final_merge.apply(lambda row: has_midnight(row['departure_date'], row['estimated_arrival']), axis=1)
        final_merge = final_merge.drop(columns=['date_x','date_y']).drop_duplicates().dropna()
        return final_merge

    def upsert_finaldf(self, fs, final_df):
       try:
        # Ensure unique_id is cast to int32 to avoid bigint type mismatch
        final_df['unique_id'] = final_df['unique_id'].astype('int32')  # Cast to int32
        
        # Ensure accident is cast to float64 to match expected double type in feature group
        final_df['accident'] = final_df['accident'].astype('float64')  # Cast to float64
        
        # Print the final DataFrame to inspect it
        print("Final DataFrame before upsert:")
        print(final_df)

        # Attempt to get the existing feature group
        fg = fs.get_feature_group(name="final_df_feature_group", version=1)
        print("Feature group exists. Upserting data...")
        
        try:
            fg.insert(final_df, write_options={"upsert": True})
        except hsfs.client.exceptions.RestAPIError as e:
            if "Parallel executions quota reached" in str(e):
                print("Max parallel executions reached. Retrying after a pause...")
                time.sleep(60)  # Wait for some time before retrying
                fg.insert(final_df, write_options={"upsert": True})
            else:
                print(f"Error inserting data into feature group: {e}")
                
       except hsfs.client.exceptions.FeatureStoreException as e:
        # Handle FeatureStoreException directly
        print(f"Error retrieving feature group: {e}")
        if 'Feature group not found' in str(e):
            print("Feature group not found. Creating a new feature group...")
            new_fg = fs.create_feature_group(
                name="final_df_feature_group",
                version=1,
                primary_key=['unique_id'],
                description="Feature group for Final Data",
            )
            new_fg.insert(final_df)
            print("Feature group created and data inserted.")
        else:
            print(f"Error in feature group operations: {e}")'''


'''from datetime import datetime
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import DateTime, create_engine, MetaData, Table, Column, Integer, String, text, inspect
from sqlalchemy.dialects.postgresql import insert
from io import StringIO
import configparser
import sys
import configparser
config = configparser.RawConfigParser()
import hsfs

# Define the path to the configuration file
CONFIG_FILE_PATH = r'C:\Desktop\Truck Project\src\config\config.ini'
from src.config import *

class FeatureEngineering:
    def _init_(self):
        self.config = configparser.RawConfigParser()
        self.config = config.read(CONFIG_FILE_PATH)

    def read_data_from_hopsworks(self,feature_store,feature_group_names):
        try:
            # Dictionary to store DataFrames for each feature group
            feature_group_dataframes = {}

            for name in feature_group_names:
                try:
                    # Get feature group by name and version
                    fg_metadata = feature_store.get_feature_group(name, version=1)  # Change version as needed

                    # If fg_metadata is already a DataFrame, use it directly
                    if isinstance(fg_metadata, pd.DataFrame):
                        fg_df = fg_metadata
                    else:
                        fg_df = fg_metadata.read()  # If it's not, read it

                    # Store the DataFrame with the feature group's name
                    feature_group_dataframes[name] = fg_df
                except Exception as e:
                    print(f"Error reading feature group {name}: {e}")

            return feature_group_dataframes

        except Exception as e:
            print(f"Error retrieving feature groups: {e}")
            return 
        
        
    def remove_columns(self,dataframes, columns_to_remove):
    
        for df_name, df in dataframes.items():
            # Drop specified columns if they exist in the DataFrame
            df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
            print(f"Removed columns {columns_to_remove} from {df_name}. Remaining columns: {df.columns.tolist()}")

        return dataframes 
      
    def remove_duplicates(self,duplicate_columns,dataframes):
        for df_name, columns in duplicate_columns.items():
            if df_name in dataframes:
                dataframes[df_name] = dataframes[df_name].drop_duplicates(subset=columns)
                print(f"Dropped duplicates from {df_name} based on {columns}")
                
        return dataframes
    
    
    def custom_mode(self,x):
        return x.mode().iloc[0] if not x.mode().empty else np.nan
 
    def getting_schedule_weather_data(self,schedule_df_cleaned,route_weather_df_cleaned):
        
        # Round 'estimated_arrival' and 'departure_date' to the nearest 6 hours
        schedule_df_cleaned['estimated_arrival'] = schedule_df_cleaned['estimated_arrival'].dt.ceil('6H')
        schedule_df_cleaned['departure_date'] = schedule_df_cleaned['departure_date'].dt.floor('6H')

            # Create date ranges between 'departure_date' and 'estimated_arrival'
        schedule_df_cleaned['date'] = [pd.date_range(start=row['departure_date'], end=row['estimated_arrival'], freq='6H').tolist() for _, row in schedule_df_cleaned.iterrows()]

            # Explode the 'date' column to create separate rows
        expanded_schedule_df = schedule_df_cleaned.explode('date').reset_index(drop=True)

            # Merge schedule with weather data on 'route_id' and 'date'
        schedule_weather_df = pd.merge(expanded_schedule_df, route_weather_df_cleaned, on=['route_id', 'date'], how='left', validate="many_to_many")
        schedule_weather_df = schedule_weather_df.drop_duplicates()
            # Group by 'unique_id', 'truck_id', 'route_id' and aggregate weather data
        try:
                schedule_weather_grp = schedule_weather_df.groupby(['truck_id', 'route_id'], as_index=False).agg(
                    route_avg_temp=('temp', 'mean'),
                    route_avg_wind_speed=('wind_speed', 'mean'),
                    route_avg_precip=('precip', 'mean'),
                    route_avg_humidity=('humidity', 'mean'),
                    route_avg_visibility=('visibility', 'mean'),
                    route_avg_pressure=('pressure', 'mean'),
                    route_description=('description', self.custom_mode)
                )
        except Exception as e:
                print(f"Error during groupby: {e}")
                schedule_weather_grp = pd.DataFrame()

        # print("Columns in schedule_df_cleaned:", schedule_df_cleaned.columns.tolist())

        # Merge the grouped data back to the original schedule_df
        schedule_weather_merge = pd.merge(expanded_schedule_df,schedule_weather_grp, on=['truck_id', 'route_id'], how='left', validate="many_to_many")
        schedule_weather_merge = schedule_weather_merge.drop_duplicates()
        schedule_weather_merge = schedule_weather_merge.dropna()
        return schedule_weather_merge
    
    def nearest_hour_schedule_route(self,truck_schedule_df,route_df_cleaned):
        nearest_hour_schedule_df = truck_schedule_df.copy()
        nearest_hour_schedule_df['estimated_arrival_nearest_hour'] = nearest_hour_schedule_df['estimated_arrival'].dt.round("H")
        nearest_hour_schedule_df['departure_date_nearest_hour'] = nearest_hour_schedule_df['departure_date'].dt.round("H")
        nearest_hour_schedule_route_df = pd.merge(nearest_hour_schedule_df, route_df_cleaned, on='route_id', how='left', validate="many_to_many")
        for col in nearest_hour_schedule_route_df.columns:
            if isinstance(nearest_hour_schedule_route_df[col].iloc[0], list):
                nearest_hour_schedule_route_df[col] = nearest_hour_schedule_route_df[col].apply(lambda x: ', '.join(map(str, x)))
        nearest_hour_schedule_route_df=nearest_hour_schedule_route_df.dropna()
        nearest_hour_schedule_route_df=nearest_hour_schedule_route_df.drop_duplicates()
        return nearest_hour_schedule_route_df,nearest_hour_schedule_df
    
    def origin_weather_data(self,weather_df_cleaned,nearest_hour_schedule_route_df):
                # Prepare origin and destination weather dataframes (Assume weather_data defined)
        origin_weather_data = weather_df_cleaned.copy()
        # Assuming 'origin_id' in nearest_hour_schedule_route_df maps to 'city_id' in origin_weather_data
        origin_weather_merge = pd.merge(nearest_hour_schedule_route_df,origin_weather_data,left_on=['origin_id','departure_date_nearest_hour'], right_on=['city_id','datetime'],how='left', validate="many_to_many")
        origin_weather_merge = origin_weather_merge.drop(columns='datetime')
        origin_weather_merge=origin_weather_merge.dropna()
        origin_weather_merge=origin_weather_merge.drop_duplicates()
        return origin_weather_merge
    
    def origin_destination_weather(self,origin_weather_merge_clean,weather_df_cleaned):
        destination_weather_data = weather_df_cleaned.copy()
        origin_destination_weather_merge = pd.merge(origin_weather_merge_clean,destination_weather_data,left_on=['destination_id','estimated_arrival_nearest_hour'],
        right_on=['city_id','datetime'], how='left',suffixes=('_origin', '_destination'), validate="many_to_many")
        origin_destination_weather_merge = origin_destination_weather_merge.drop(columns=['datetime','city_id_origin','city_id_destination'])
        origin_destination_weather_merge=origin_destination_weather_merge.drop_duplicates()
        origin_destination_weather_merge=origin_destination_weather_merge.dropna()
        return origin_destination_weather_merge
    
    def scheduled_route_traffic(self,nearest_hour_schedule_df,traffic_df_clean):
        hourly_exploded_scheduled_df = (
            nearest_hour_schedule_df.assign(custom_date=[pd.date_range(start, end, freq='H') for start, end in zip(nearest_hour_schedule_df['departure_date'], nearest_hour_schedule_df['estimated_arrival'])])
            .explode('custom_date', ignore_index=True)
        )

        scheduled_traffic = hourly_exploded_scheduled_df.merge(traffic_df_clean, left_on=['route_id','custom_date'],right_on=['route_id','datetime'], how='left')
        def custom_agg(values):
            return int(any(values == 1))

        scheduled_route_traffic = scheduled_traffic.groupby(['truck_id', 'route_id'], as_index=False).agg(
            avg_no_of_vehicles=('no_of_vehicles', 'mean'),
            accident=('accident', custom_agg)
        )
        scheduled_route_traffic=scheduled_route_traffic.drop_duplicates()
        return scheduled_route_traffic
    
    def final_merge(self,origin_destination_weather_df,scheduled_route_traffic,schedule_weather_merge,trucks_df_cleaned,drivers_df_cleaned):
        origin_destination_weather_traffic_merge = origin_destination_weather_df.merge(scheduled_route_traffic, on=['truck_id', 'route_id'], how='left')
        origin_destination_weather_traffic_merge=origin_destination_weather_traffic_merge.drop_duplicates()
        # Final merge with traffic data
        merged_data_weather_traffic = pd.merge(schedule_weather_merge, origin_destination_weather_traffic_merge, on=['truck_id', 'route_id','delay','departure_date','estimated_arrival'], how='left', validate="many_to_many")
        merged_data_weather_traffic_trucks = pd.merge(merged_data_weather_traffic, trucks_df_cleaned, on='truck_id', how='left', validate="many_to_many")
        final_merge = pd.merge(merged_data_weather_traffic_trucks, drivers_df_cleaned, left_on='truck_id', right_on='vehicle_no', how='left', validate="many_to_many")
        def has_midnight(start, end):
            return int(start.date() != end.date())
        final_merge['is_midnight'] = final_merge.apply(lambda row: has_midnight(row['departure_date'], row['estimated_arrival']), axis=1)
        final_merge = final_merge.drop(columns=['date_x','date_y'])
        final_merge=final_merge.drop_duplicates()
        final_merge=final_merge.dropna()
        return final_merge
    
    
    def upsert_finaldf(self,fs,final_df):
        try:
            # Attempt to get the existing feature group
            fg = fs.get_feature_group(name="final_df_feature_group", version=1)
            print("Feature group  exists. Upserting data...")
            
            try:
                fg.insert(final_df, write_options={"upsert": True})
            except hsfs.client.exceptions.RestAPIError as e:
                if "Parallel executions quota reached" in str(e):
                    print("Max parallel executions reached. Retrying after a pause...")
                    time.sleep(60)  # Wait for some time before retrying
                    fg.insert(final_df, write_options={"upsert": True})
                else:
                    print(f"Error inserting data into feature group : {e}")
                    
        except hsfs.client.exceptions.RestAPIError as e:
            error_body = json.loads(e.response.text)  # Assuming e.response.text contains the error JSON
            error_code = error_body.get('errorCode')
            
            if error_code == 270009:  # Feature group not found
                print("Feature group not found. Creating a new feature group...")
                new_fg = fs.create_feature_group(
                    name="final_df_feature_group",
                    version=1,
                    primary_key=['unique_id'],
                    description="Feature group for Final Data",
                    # online_enabled=True  # Enable online feature store if needed
                )
                new_fg.insert(final_df)
                print("Feature group created and data inserted.")
            else:
                print(f"Error retrieving feature group : {e}")'''