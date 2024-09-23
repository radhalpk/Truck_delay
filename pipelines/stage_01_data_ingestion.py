import os
import sys
import configparser
import pandas as pd
import requests
import io

# Add the src directory to the Python path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

# Import the ingestion component
from src.components.data_ingestion import PostgreSQLIngestion

def read_config(config_file=None):
    if config_file is None:
        # Construct the path to config.ini relative to the script location
        config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'config', 'config.ini'))
    
    print(f"Attempting to read config file at: {config_file}")
    config = configparser.ConfigParser()
    if not os.path.isfile(config_file):
        print(f"Config file not found: {config_file}")
        return None
    config.read(config_file)
    print("Config sections:", config.sections())
    return config

def download_github_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))
    else:
        print(f"Failed to download data from {url}")
        return None

def run_ingestion_pipeline():
    print("Current working directory:", os.getcwd())
    config = read_config()

    if config is None:
        print("Failed to read config file.")
        return

    db_config = {
        'host': config['POSTGRESQL']['host'].strip(),
        'port': int(config['POSTGRESQL']['port'].strip()),
        'dbname': config['POSTGRESQL']['dbname'].strip(),
        'user': config['POSTGRESQL']['user'].strip(),
        'password': config['POSTGRESQL']['password'].strip()
    }

    ingestion = PostgreSQLIngestion(db_config)
    ingestion.connect()

    table_dataframes = {}  # Dictionary to store dataframes

    for key in config['GITHUB_DATA']:
        data = download_github_data(config['GITHUB_DATA'][key].strip())
        if data is not None:
            table_name = key.replace('_url', '')
            ingestion.upsert_data(data, table_name)

    tables = ingestion.get_table_names()

    if tables:
        for table in tables:
            data = ingestion.fetch_data(table)
            if data is not None:
                table_dataframes[table] = data  # Store dataframe in dictionary
                print(f"Data from table {table} loaded into memory.")

    ingestion.close()

    return table_dataframes  # Return the dictionary containing all dataframes

if __name__ == "__main__":
    try:
        print("Ingestion pipeline started")
        data_frames = run_ingestion_pipeline()
        print("Ingestion pipeline ran successfully.")
        # Example usage of data_frames
        print(data_frames.keys())  # To list all table names which have been converted to dataframes
    except Exception as e:
        print(f"Failed to complete the ingestion pipeline: {str(e)}")