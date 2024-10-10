import configparser
import os.path as path
import pandas as pd
import sys
import os
import hopsworks

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.components.data_merging import FeatureEngineering

STAGE_NAME = "Feature Engineering"

# Initialize Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

class FeatureEngineeringPipeline:
    def __init__(self):
        self.feature_obj = FeatureEngineering()
        self.feature_group_names = [
            "city_weather_features", 
            "drivers_features", 
            "routes_features", 
            "trucks_features", 
            "truck_schedule_features", 
            "traffic_table_features", 
            "routes_weather_features"
        ]
        self.duplicate_columns = {
            'city_weather_features': ['city_id', 'datetime'],
            'trucks_features': ['truck_id'],  
            'truck_schedule_features': ['truck_id', 'route_id', 'departure_date'],  
            'drivers_features': ['driver_id'],
            'routes_features': ['route_id', 'destination_id', 'origin_id'],
            'routes_weather_features': ['route_id', 'date']
        }
        self.columns_to_remove = ['index', 'event_date']

    def main(self):
        try:
            # Retrieve dataframes from Hopsworks
            dataframes = self.feature_obj.read_data_from_hopsworks(fs, self.feature_group_names)
            print(dataframes['truck_schedule_features'])

            # Remove duplicates based on the specified columns
            dataframes = self.feature_obj.remove_duplicates(self.duplicate_columns, dataframes)

            # Remove unnecessary columns ('index', 'event_date')
            dataframes = self.feature_obj.remove_columns(dataframes, self.columns_to_remove)

            # Merge weather data with truck schedule
            schedule_weather_merge = self.feature_obj.getting_schedule_weather_data(
                dataframes['truck_schedule_features'], dataframes['routes_weather_features']
            )

            # Nearest hour schedule route merging
            nearest_hour_schedule_route_df, nearest_hour_schedule_df = self.feature_obj.nearest_hour_schedule_route(
                dataframes['truck_schedule_features'], dataframes['routes_features']
            )

            # Merge origin weather data
            origin_weather_merge = self.feature_obj.origin_weather_data(
                dataframes['city_weather_features'], nearest_hour_schedule_route_df
            )

            # Merge destination weather data
            origin_destination_weather_df = self.feature_obj.origin_destination_weather(
                origin_weather_merge, dataframes['city_weather_features']
            )

            # Traffic data merging
            scheduled_route_traffic = self.feature_obj.scheduled_route_traffic(
                nearest_hour_schedule_df, dataframes['traffic_table_features']
            )

            # Final merge of all the data
            final_df = self.feature_obj.final_merge(
                origin_destination_weather_df,
                scheduled_route_traffic,
                schedule_weather_merge,
                dataframes['trucks_features'],
                dataframes['drivers_features']
            )

            # Add and reorder 'unique_id' column
            final_df['unique_id'] = range(1, len(final_df) + 1)  # Starting from 1
            cols = ['unique_id'] + [col for col in final_df.columns if col != 'unique_id']  # Place 'unique_id' at the start
            final_df = final_df[cols]  # Reorder the DataFrame

            # Ensure 'unique_id' is an integer
            final_df['unique_id'] = final_df['unique_id'].astype(int)

            # Reorder the 'delay' column to appear last
            delay_col = final_df.pop('delay')  # Remove the 'delay' column
            final_df['delay'] = delay_col      # Append the 'delay' column to the end

            print(final_df.columns)
            print(final_df.shape)
            print(final_df)

            # Upsert final DataFrame to Hopsworks
            self.feature_obj.upsert_finaldf(fs, final_df)

        except Exception as e:
            print(f"Error in Feature Engineering Pipeline: {e}")
            raise e

if __name__ == '__main__':    
    try:
        print(">>>>>> Stage started <<<<<< :", STAGE_NAME)
        obj = FeatureEngineeringPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e
