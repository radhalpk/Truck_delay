import os
import sys
import pandas as pd
from sqlalchemy import inspect

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the DataClean class from src.components
from src.components.data_cleaning import DataClean  # Import your DataClean class

# Configuration file path
CONFIG_FILE_PATH = os.path.join(project_root, 'src', 'config', 'config.ini')

class DataCleaningPipeline:
    def __init__(self):
        # Initialize the DataClean object with the config path
        self.cleaning = DataClean(CONFIG_FILE_PATH)

    def get_all_tables(self):
        """Get all table names from the PostgreSQL database."""
        inspector = inspect(self.cleaning.engine)
        return inspector.get_table_names()

    def process_table(self, df, table_name):
        """Process each table based on its specific cleaning requirements."""
        # Step 1: Sanitize column names
        df = self.cleaning.sanitize_dataframe(df)

        # Step 2: Specific processing based on the table name
        if 'city_weather' in table_name or 'route_weather' in table_name:
            df = self.cleaning.merge_date_hour(df)  # Merge date and hour
            df = self.cleaning.remove_outliers_iqr(df)  # Remove outliers
        elif 'drivers' in table_name:
            df = self.cleaning.drop_duplicates(df)  # Drop duplicates
            df = self.cleaning.remove_outliers_iqr(df)  # Remove outliers
        elif 'routes' in table_name:
            df = self.cleaning.remove_outliers_iqr(df)  # Remove outliers
        elif 'truck_schedule' in table_name:
            df = self.cleaning.drop_duplicates(df)  # Drop duplicates

        # Further table-specific cleaning can be added here
        return df

    def run_pipeline(self):
        """Run the entire data cleaning pipeline."""
        try:
            # Step 1: Get all tables from the database
            tables = self.get_all_tables()
            for table_name in tables:
                print(f"Processing table: {table_name}")
                
                # Step 2: Read table into a DataFrame
                df = self.cleaning.read_table(table_name)

                if df.empty:
                    print(f"Skipping empty table: {table_name}")
                    continue

                # Step 3: Process the table for cleaning
                df = self.process_table(df, table_name)

                # Step 4: Add event_date and index_column
                df = self.cleaning.add_event_date(df)
                df = self.cleaning.add_index_column(df)

                # Step 5: Save the cleaned DataFrame to Hopsworks
                self.cleaning.save_cleaned_data_hopsworks(df, table_name)
            
            print("Data cleaning pipeline completed successfully.")

        except Exception as e:
            print(f"Error in pipeline execution: {e}")
            raise e

if __name__ == '__main__':
    try:
        STAGE_NAME = "DATACLEANING"
        print(f">>>>>> Stage started: {STAGE_NAME} <<<<<<")
        pipeline = DataCleaningPipeline()
        pipeline.run_pipeline()
        print(f">>>>>> Stage completed: {STAGE_NAME} <<<<<<")
    except Exception as e:
        print(f"An error occurred in stage: {STAGE_NAME}")
        print(e)
