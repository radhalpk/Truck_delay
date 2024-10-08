import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# cleaning_pipeline.py
from src.components.data_cleaning import DataCleaning
# Configuration file path
CONFIG_FILE_PATH = r'C:\Desktop\Truck Project\src\config\config.ini'

class DataCleaningPipeline:
    def __init__(self):
        self.cleaning = DataCleaning(CONFIG_FILE_PATH)  # Initialize the cleaning object

    def run_pipeline(self):
        """Run the full data pipeline."""
        try:
            # Step 1: Fetch tables from PostgreSQL
            self.cleaning.read_tables_as_dataframes()

            # Step 2: Clean the data
            self.cleaning.clean_dataframes()

            # Step 3: Connect to Hopsworks
            fs = self.cleaning.connect_to_hopsworks()

            # Step 4: Upload cleaned data to Hopsworks feature store
            self.cleaning.upload_to_hopsworks(fs)

        finally:
            # Step 5: Close PostgreSQL connection
            self.cleaning.close_postgresql_connection()

# Example usage
if __name__ == '__main__':
    try:
        STAGE_NAME = "DATACLEANING"
        print(">>>>>> Stage started <<<<<< :", STAGE_NAME)
        
        # Initialize the pipeline
        pipeline = DataCleaningPipeline()
        
        # Run the full pipeline
        pipeline.run_pipeline()
        
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(f"Error occurred in stage: {STAGE_NAME}")
        print(e)
        raise e

