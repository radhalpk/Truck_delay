import os
import sys

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

from src.components.data_merging import DataFetcher, DataPreparation

if __name__ == "__main__":
    try:
        STAGE_NAME = "DATA_PREPARATION"
        print(f">>>>>> Stage started: {STAGE_NAME} <<<<<<")

        # Step 1: Fetch all data
        fetcher = DataFetcher()
        dataframes = fetcher.fetch_all_data()

        # Step 2: Prepare the data
        preparer = DataPreparation()
        prepared_dataframes = preparer.prepare_data(dataframes)

        # Output some information for each dataframe
        for name, df in prepared_dataframes.items():
            print(f"{name} has {len(df)} records after preparation.")

        print(f">>>>>> Stage completed: {STAGE_NAME} <<<<<<")

    except Exception as e:
        print(f"Error during {STAGE_NAME}: {e}")
        raise e
