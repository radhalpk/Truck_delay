import pandas as pd
import numpy as np
import hsfs
import configparser
import time

# Config file path
CONFIG_FILE_PATH = 'C:/Desktop/Truck Project/src/config/config.ini'

class FeatureEngineering:
    def __init__(self):
        self.config = configparser.RawConfigParser()
        self.config.read(CONFIG_FILE_PATH)

    def read_data_from_hopsworks(self, feature_store, feature_group_names):
        """Fetch feature groups from Hopsworks Feature Store."""
        try:
            feature_group_dataframes = {}
            for name in feature_group_names:
                try:
                    # Get feature group by name and version
                    fg_metadata = feature_store.get_feature_group(name, version=1)
                    # Read the data from feature group
                    fg_df = fg_metadata.read()
                    # Store the DataFrame
                    feature_group_dataframes[name] = fg_df
                except Exception as e:
                    print(f"Error reading feature group {name}: {e}")
            return feature_group_dataframes
        except Exception as e:
            print(f"Error retrieving feature groups: {e}")
            return 

    def remove_columns(self, dataframes, columns_to_remove):
        """Remove unnecessary columns from the DataFrames."""
        for df_name, df in dataframes.items():
            df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
            print(f"Removed columns {columns_to_remove} from {df_name}. Remaining columns: {df.columns.tolist()}")
        return dataframes

    def remove_duplicates(self, duplicate_columns, dataframes):
        """Remove duplicate rows from DataFrames based on the specified subset columns."""
        for df_name, columns in duplicate_columns.items():
            if df_name in dataframes:
                dataframes[df_name] = dataframes[df_name].drop_duplicates(subset=columns)
                print(f"Dropped duplicates from {df_name} based on {columns}")
        return dataframes

    def custom_mode(self, x):
        """Helper function to get mode (most frequent value) or NaN if empty."""
        return x.mode().iloc[0] if not x.mode().empty else np.nan

    def getting_schedule_weather_data(self, schedule_df_cleaned, route_weather_df_cleaned):
        """Merge truck schedule and weather data on 'route_id' and 'date'."""
        schedule_df_cleaned['estimated_arrival'] = pd.to_datetime(schedule_df_cleaned['estimated_arrival'], errors='coerce').dt.ceil('6H')
        schedule_df_cleaned['departure_date'] = pd.to_datetime(schedule_df_cleaned['departure_date'], errors='coerce').dt.floor('6H')
        schedule_df_cleaned['date'] = [pd.date_range(start=row['departure_date'], end=row['estimated_arrival'], freq='6H').tolist() for _, row in schedule_df_cleaned.iterrows()]
        expanded_schedule_df = schedule_df_cleaned.explode('date').reset_index(drop=True)
        schedule_weather_df = pd.merge(expanded_schedule_df, route_weather_df_cleaned, on=['route_id', 'date'], how='left', validate="many_to_many")
        schedule_weather_df = schedule_weather_df.drop_duplicates()
        schedule_weather_grp = schedule_weather_df.groupby(['truck_id', 'route_id'], as_index=False).agg(
            route_avg_temp=('temp', 'mean'),
            route_avg_wind_speed=('wind_speed', 'mean'),
            route_avg_precip=('precip', 'mean'),
            route_avg_humidity=('humidity', 'mean'),
            route_avg_visibility=('visibility', 'mean'),
            route_avg_pressure=('pressure', 'mean'),
            route_description=('description', self.custom_mode)
        )
        schedule_weather_merge = pd.merge(expanded_schedule_df, schedule_weather_grp, on=['truck_id', 'route_id'], how='left', validate="many_to_many")
        schedule_weather_merge = schedule_weather_merge.drop_duplicates()
        return schedule_weather_merge

    def nearest_hour_schedule_route(self, truck_schedule_df, route_df_cleaned):
        """Round 'estimated_arrival' and 'departure_date' to the nearest hour and merge with route data."""
        nearest_hour_schedule_df = truck_schedule_df.copy()
        nearest_hour_schedule_df['estimated_arrival_nearest_hour'] = pd.to_datetime(nearest_hour_schedule_df['estimated_arrival'], errors='coerce').dt.round("H")
        nearest_hour_schedule_df['departure_date_nearest_hour'] = pd.to_datetime(nearest_hour_schedule_df['departure_date'], errors='coerce').dt.round("H")
        nearest_hour_schedule_route_df = pd.merge(nearest_hour_schedule_df, route_df_cleaned, on='route_id', how='left', validate="many_to_many")
        nearest_hour_schedule_route_df = nearest_hour_schedule_route_df.dropna().drop_duplicates()
        return nearest_hour_schedule_route_df, nearest_hour_schedule_df

    def origin_weather_data(self, weather_df_cleaned, nearest_hour_schedule_route_df):
        """Merge weather data for origin cities."""
        # Check if 'datetime' column exists
        if 'datetime' not in weather_df_cleaned.columns:
            print("Warning: 'datetime' column not found in weather data. Ensure correct column exists.")
            return pd.DataFrame()  # Return empty DataFrame in case of missing column

        origin_weather_merge = pd.merge(nearest_hour_schedule_route_df, weather_df_cleaned, left_on=['origin_id', 'departure_date_nearest_hour'], right_on=['city_id', 'datetime'], how='left', validate="many_to_many")
        origin_weather_merge = origin_weather_merge.drop(columns='datetime').dropna().drop_duplicates()
        return origin_weather_merge

    def origin_destination_weather(self, origin_weather_merge_clean, weather_df_cleaned):
        """Merge weather data for both origin and destination cities."""
        origin_destination_weather_merge = pd.merge(origin_weather_merge_clean, weather_df_cleaned, left_on=['destination_id', 'estimated_arrival_nearest_hour'], right_on=['city_id', 'datetime'], how='left', suffixes=('_origin', '_destination'), validate="many_to_many")
        origin_destination_weather_merge = origin_destination_weather_merge.drop(columns=['datetime', 'city_id_origin', 'city_id_destination']).dropna().drop_duplicates()
        return origin_destination_weather_merge

    def scheduled_route_traffic(self, nearest_hour_schedule_df, traffic_df_clean):
        """Merge traffic data with the truck schedule."""
        # Ensure 'datetime' column exists in traffic data
        if 'datetime' not in traffic_df_clean.columns:
            print("Error: 'datetime' column not found in traffic data.")
            return pd.DataFrame()  # Return empty DataFrame if column is missing

        traffic_df_clean['datetime'] = pd.to_datetime(traffic_df_clean['datetime'], errors='coerce')

        hourly_exploded_scheduled_df = nearest_hour_schedule_df.assign(custom_date=[pd.date_range(start, end, freq='H') for start, end in zip(nearest_hour_schedule_df['departure_date'], nearest_hour_schedule_df['estimated_arrival'])]).explode('custom_date', ignore_index=True)
        scheduled_traffic = pd.merge(hourly_exploded_scheduled_df, traffic_df_clean, left_on=['route_id', 'custom_date'], right_on=['route_id', 'datetime'], how='left')
        scheduled_route_traffic = scheduled_traffic.groupby(['truck_id', 'route_id'], as_index=False).agg(
            avg_no_of_vehicles=('no_of_vehicles', 'mean'),
            accident=('accident', lambda x: int(any(x == 1)))
        )
        return scheduled_route_traffic

    def final_merge(self, origin_destination_weather_df, scheduled_route_traffic, schedule_weather_merge, trucks_df_cleaned, drivers_df_cleaned):
        """Final merge of all data into a single DataFrame."""
        origin_destination_weather_traffic_merge = pd.merge(origin_destination_weather_df, scheduled_route_traffic, on=['truck_id', 'route_id'], how='left')
        merged_data_weather_traffic = pd.merge(schedule_weather_merge, origin_destination_weather_traffic_merge, on=['truck_id', 'route_id', 'delay', 'departure_date', 'estimated_arrival'], how='left', validate="many_to_many")
        merged_data_weather_traffic_trucks = pd.merge(merged_data_weather_traffic, trucks_df_cleaned, on='truck_id', how='left', validate="many_to_many")
        final_merge = pd.merge(merged_data_weather_traffic_trucks, drivers_df_cleaned, left_on='truck_id', right_on='vehicle_no', how='left', validate="many_to_many")
        final_merge['is_midnight'] = final_merge.apply(lambda row: int(row['departure_date'].date() != row['estimated_arrival'].date()), axis=1)
        final_merge = final_merge.drop_duplicates().dropna()
        return final_merge

    def upsert_finaldf(self, fs, final_df):
        """Upsert the final DataFrame to Hopsworks Feature Store."""
        try:
            fg = fs.get_feature_group(name="final_df_feature_group", version=1)
            fg.insert(final_df, write_options={"upsert": True})
        except hsfs.client.exceptions.RestAPIError as e:
            if "Parallel executions quota reached" in str(e):
                print("Max parallel executions reached. Retrying after a pause...")
                time.sleep(60)
                fg.insert(final_df, write_options={"upsert": True})
            else:
                print(f"Error inserting data into feature group: {e}")
        except Exception as e:
            print(f"Error retrieving feature group: {e}")
