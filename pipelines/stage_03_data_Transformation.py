'''import configparser
import pandas as pd
import sys
import os
import hopsworks
import hsfs

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.components.data_merging import FeatureEngineering
STAGE_NAME = "Data Merging"
project = hopsworks.login()
fs = project.get_feature_store()

class FeatureEngineeringPipeline:
    def __init__(self):
        self.feature_obj = FeatureEngineering()
        self.feature_group_names = ["city_weather", "drivers_table", "routes_table", "trucks_table", "truck_schedule_table", "traffic_table", "routes_weather"]
        self.duplicate_columns = {
            'city_weather': ['city_id', 'datetime'],
            'trucks_table': ['truck_id'],  
            'truck_schedule_table': ['truck_id', 'route_id', 'departure_date'],  
            'drivers_table': ['driver_id'],
            'routes_table': ['route_id', 'destination_id', 'origin_id'],
            'routes_weather': ['route_id', 'date']
        }
        self.columns_to_remove = ['id', 'event_time']

    def main(self):
        try:
            # Read data from the feature store
            dataframes = self.feature_obj.read_data_from_hopsworks(fs, self.feature_group_names)
            dataframes = self.feature_obj.remove_duplicates(self.duplicate_columns, dataframes)
            dataframes = self.feature_obj.remove_columns(dataframes, self.columns_to_remove)

            # Generate necessary dataframes for merging
            schedule_weather_merge = self.feature_obj.getting_schedule_weather_data(dataframes['truck_schedule_table'], dataframes['routes_weather'])
            nearest_hour_schedule_route_df, nearest_hour_schedule_df = self.feature_obj.nearest_hour_schedule_route(dataframes['truck_schedule_table'], dataframes['routes_table'])
            origin_weather_merge = self.feature_obj.origin_weather_data(dataframes['city_weather'], nearest_hour_schedule_route_df)
            origin_destination_weather_df = self.feature_obj.origin_destination_weather(origin_weather_merge, dataframes['city_weather'])
            scheduled_route_traffic = self.feature_obj.scheduled_route_traffic(nearest_hour_schedule_df, dataframes['traffic_table'])

            # Final merge
            final_df = self.feature_obj.final_merge(origin_destination_weather_df, scheduled_route_traffic, schedule_weather_merge, dataframes['trucks_table'], dataframes['drivers_table'])

            # Ensure unique_id and accident have the right types
            if 'unique_id' not in final_df.columns:
                final_df['unique_id'] = range(1, len(final_df) + 1)
            final_df['unique_id'] = final_df['unique_id'].astype('int64')
            final_df['accident'] = final_df['accident'].astype('int64')

            # Move 'delay' to the end
            delay_col = final_df.pop('delay')
            final_df['delay'] = delay_col

            # Check if the feature group exists; if not, create it
            fg_name = "final_df_feature_group"
            try:
                fg = fs.get_feature_group(name=fg_name, version=1)
                print(f"Feature group {fg_name} exists. Proceeding with upsert.")
            except hsfs.client.exceptions.FeatureGroupNotFound:
                print(f"Feature group {fg_name} not found. Creating new feature group...")
                fg = fs.create_feature_group(
                    name=fg_name,
                    version=1,
                    primary_key=['unique_id'],
                    description="Feature group for Final Data with int64 unique_id and accident"
                )

            # Upsert the data into the feature group
            fg.insert(final_df, write_options={"upsert": True})
            print(f"Data successfully upserted to {fg_name}.")

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
        raise e'''



'''import configparser
import pandas as pd
import sys
import os
import hopsworks

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.components.data_merging import FeatureEngineering
STAGE_NAME = "Data Merging"
project = hopsworks.login()
fs = project.get_feature_store()

class FeatureEngineeringPipeline:
    def __init__(self):
        self.feature_obj = FeatureEngineering()
        self.feature_group_names = ["city_weather", "drivers_table", "routes_table", "trucks_table", "truck_schedule_table", "traffic_table", "routes_weather"]
        self.duplicate_columns = {
            'city_weather': ['city_id', 'datetime'],
            'trucks_table': ['truck_id'],  
            'truck_schedule_table': ['truck_id', 'route_id', 'departure_date'],  
            'drivers_table': ['driver_id'],
            'routes_table': ['route_id', 'destination_id', 'origin_id'],
            'routes_weather': ['route_id', 'date']
        }
        self.columns_to_remove = ['id', 'event_time']

    def main(self):
        try:
            dataframes = self.feature_obj.read_data_from_hopsworks(fs, self.feature_group_names)
            dataframes = self.feature_obj.remove_duplicates(self.duplicate_columns, dataframes)
            dataframes = self.feature_obj.remove_columns(dataframes, self.columns_to_remove)

            schedule_weather_merge = self.feature_obj.getting_schedule_weather_data(dataframes['truck_schedule_table'], dataframes['routes_weather'])
            nearest_hour_schedule_route_df, nearest_hour_schedule_df = self.feature_obj.nearest_hour_schedule_route(dataframes['truck_schedule_table'], dataframes['routes_table'])
            origin_weather_merge = self.feature_obj.origin_weather_data(dataframes['city_weather'], nearest_hour_schedule_route_df)
            origin_destination_weather_df = self.feature_obj.origin_destination_weather(origin_weather_merge, dataframes['city_weather'])
            scheduled_route_traffic = self.feature_obj.scheduled_route_traffic(nearest_hour_schedule_df, dataframes['traffic_table'])

            final_df = self.feature_obj.final_merge(origin_destination_weather_df, scheduled_route_traffic, schedule_weather_merge, dataframes['trucks_table'], dataframes['drivers_table'])

            #final_df['unique_id'] = range(1, len(final_df) + 1)
            #cols = ['unique_id'] + [col for col in final_df.columns if col != 'unique_id']
            #final_df = final_df[cols]
            #final_df['unique_id'] = final_df['unique_id'].astype(int)
            
            delay_col = final_df.pop('delay')
            final_df['delay'] = delay_col

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
        raise e'''


'''import configparser
import os.path as path
import pandas as pd
import sys
import os
import hopsworks

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.components.data_merging import FeatureEngineering


STAGE_NAME = "Data Merging"

# Initialize Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

class FeatureEngineeringPipeline:
    def __init__(self):
        self.feature_obj = FeatureEngineering()
        self.feature_group_names = ["city_weather","drivers_table","routes_table","trucks_table","truck_schedule_table","traffic_table","routes_weather"]  # Replace with your actual feature group names
        self.duplicate_columns = {
        'city_weather': ['city_id', 'datetime'],
        'trucks_table': ['truck_id'],  
        'truck_schedule_table': ['truck_id', 'route_id', 'departure_date'],  
        # 'traffic_table': ['traffic_id'],
        'drivers_table':['driver_id'],
        'routes_table':['route_id', 'destination_id', 'origin_id'],
        'routes_weather':['route_id', 'date']
        }
        self.columns_to_remove = ['id', 'event_time']

    def main(self):
        try:
            dataframes = self.feature_obj.read_data_from_hopsworks(fs,self.feature_group_names)
            print(dataframes['truck_schedule_table'])
            dataframes = self.feature_obj.remove_duplicates(self.duplicate_columns,dataframes)
            dataframes = self.feature_obj.remove_columns(dataframes, self.columns_to_remove)
            schedule_weather_merge=self.feature_obj.getting_schedule_weather_data(dataframes['truck_schedule_table'],dataframes['routes_weather'])
            nearest_hour_schedule_route_df,nearest_hour_schedule_df=self.feature_obj.nearest_hour_schedule_route(dataframes['truck_schedule_table'],dataframes['routes_table'])
            origin_weather_merge=self.feature_obj.origin_weather_data(dataframes['city_weather'],nearest_hour_schedule_route_df)
            origin_destination_weather_df=self.feature_obj.origin_destination_weather(origin_weather_merge,dataframes['city_weather'])
            scheduled_route_traffic=self.feature_obj.scheduled_route_traffic(nearest_hour_schedule_df,dataframes['traffic_table'])
            final_df=self.feature_obj.final_merge(origin_destination_weather_df,scheduled_route_traffic,schedule_weather_merge,dataframes['trucks_table'],dataframes['drivers_table'])
            final_df['unique_id'] = range(1, len(final_df) + 1)  # Starting from 1
            cols = ['unique_id'] + [col for col in final_df.columns if col != 'unique_id']  # Place 'unique_id' at the start
            final_df = final_df[cols]  # Reorder the DataFrame
            final_df['unique_id'] = final_df['unique_id'].astype(int)
            delay_col = final_df.pop('delay')  # Remove the 'delay' column
            final_df['delay'] = delay_col      # Append the 'delay' column to the end
            print(final_df.columns)
            print(final_df.shape)
            print(final_df)
            self.feature_obj.upsert_finaldf(fs,final_df)
            
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
        raise e'''
        
import configparser
import pandas as pd
import sys
import os
import hopsworks

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.components.data_merging import FeatureEngineering
CONFIG_FILE_PATH = os.path.join(project_root, 'src', 'config', 'config.ini')


STAGE_NAME = "Data Merging"
project = hopsworks.login()
fs = project.get_feature_store()

class FeatureEngineeringPipeline:
    def __init__(self):
        self.feature_obj = FeatureEngineering()
        self.feature_group_names = ["city_weather", "drivers_table", "routes_table", "trucks_table", "truck_schedule_table", "traffic_table", "routes_weather"]
        self.duplicate_columns = {
            'city_weather': ['city_id', 'datetime'],
            'trucks_table': ['truck_id'],  
            'truck_schedule_table': ['truck_id', 'route_id', 'departure_date'],  
            'drivers_table': ['driver_id'],
            'routes_table': ['route_id', 'destination_id', 'origin_id'],
            'routes_weather': ['route_id', 'date']
        }
        self.columns_to_remove = ['id', 'event_time']

    def main(self):
        try:
            # Step 1: Read data from the feature store
            print("Reading data from Hopsworks...")
            dataframes = self.feature_obj.read_data_from_hopsworks(fs, self.feature_group_names)
            
            # Step 2: Remove duplicates and unnecessary columns
            print("Removing duplicates and unnecessary columns...")
            dataframes = self.feature_obj.remove_duplicates(self.duplicate_columns, dataframes)
            dataframes = self.feature_obj.remove_columns(dataframes, self.columns_to_remove)

            # Step 3: Perform data merging
            print("Merging data...")
            schedule_weather_merge = self.feature_obj.getting_schedule_weather_data(dataframes['truck_schedule_table'], dataframes['routes_weather'])
            nearest_hour_schedule_route_df, nearest_hour_schedule_df = self.feature_obj.nearest_hour_schedule_route(dataframes['truck_schedule_table'], dataframes['routes_table'])
            origin_weather_merge = self.feature_obj.origin_weather_data(dataframes['city_weather'], nearest_hour_schedule_route_df)
            origin_destination_weather_df = self.feature_obj.origin_destination_weather(origin_weather_merge, dataframes['city_weather'])
            scheduled_route_traffic = self.feature_obj.scheduled_route_traffic(nearest_hour_schedule_df, dataframes['traffic_table'])

            # Step 4: Final merge of all data
            print("Final merge of all dataframes...")
            final_df = self.feature_obj.final_merge(origin_destination_weather_df, scheduled_route_traffic, schedule_weather_merge, dataframes['trucks_table'], dataframes['drivers_table'])

            # Step 5: Handle missing unique_id or wrong types
            if 'unique_id' not in final_df.columns:
                final_df['unique_id'] = range(1, len(final_df) + 1)
            final_df['unique_id'] = final_df['unique_id'].astype('int64')
            final_df['accident'] = final_df['accident'].astype('int64')

            # Move 'delay' column to the end of the dataframe
            delay_col = final_df.pop('delay')
            final_df['delay'] = delay_col

            # Step 6: Upsert the final dataframe to the feature store
            print("Upserting data into Hopsworks...")
            self.feature_obj.upsert_finaldf(fs, final_df)

        except Exception as e:
            print(f"Error in Feature Engineering Pipeline: {e}")
            raise e

if __name__ == '__main__':
    try:
        print(f">>>>>> Stage started: {STAGE_NAME} <<<<<<")
        obj = FeatureEngineeringPipeline()
        obj.main()
        print(f">>>>>> Stage completed: {STAGE_NAME} <<<<<<")
    except Exception as e:
        print(e)
        raise e
