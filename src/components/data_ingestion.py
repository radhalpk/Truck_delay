import os
import pandas as pd
import requests
from sqlalchemy import create_engine
import configparser
import hopsworks

# Hopsworks project login
project = hopsworks.login()
fs = project.get_feature_store()

config = configparser.RawConfigParser()  # Initialize config object

# Read configuration from the config file
CONFIG_FILE_PATH = 'C:\Desktop\Truck Project\src\config\config.ini'
STAGE_NAME = "Data Ingestion"

class DataIngestion:
    def __init__(self):
        config.read(CONFIG_FILE_PATH)
        self.username = config.get('DATABASE', 'username')
        self.password = config.get('DATABASE', 'password')
        self.host = config.get('DATABASE', 'host')
        self.port = config.get('DATABASE', 'port')
        self.database = config.get('DATABASE', 'dbname')
        
        # Set up SQLAlchemy engine to connect to PostgreSQL
        self.engine = create_engine(f'postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}')
        print(f"Connected to database: {self.database}")
        
        # Get GitHub API URL from the config file
        self.github_url = config.get('API', 'github_url')
    
    # Fetch file URLs from GitHub API
    def fetch_file_urls(self):
        response = requests.get(self.github_url)
        if response.status_code == 200:
            files = response.json()
            # Filter out only CSV files
            csv_files = [file['download_url'] for file in files if file['name'].endswith('.csv')]
            return csv_files
        else:
            raise Exception(f"Failed to fetch file URLs from GitHub API: {response.status_code}")
      
    
    # Fetch data from GitHub and store it in PostgreSQL (DBeaver)
    def fetch_and_store_data(self):
        # Fetch file URLs from GitHub API
        file_urls = self.fetch_file_urls()

        for url in file_urls:
            # Extract the table name from the CSV filename (use it as the table name)
            table_name = url.split('/')[-1].replace('.csv', '')
            print(f"Processing table: {table_name} from {url}")
            
            # Fetch data from the CSV URL
            df = pd.read_csv(url)
            
            # Store in PostgreSQL (DBeaver)
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            print(f"Stored {table_name} in the database")
            
    def load_dataframe(self, table_name):
        """
        Load a table from PostgreSQL into a pandas DataFrame
        :param table_name: Table name to fetch
        :return: pandas DataFrame
        """
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query, self.engine)   

    def fetch_data(self, query):
        """
        Fetch data from the PostgreSQL database using a SQL query.
        :param query: SQL query to fetch data
        :return: pandas DataFrame with the fetched data
        """
        with self.engine.connect() as connection:
            df = pd.read_sql(query, connection)
            return df

    def fetch_table(self, table_name):
        """
        Fetch an entire table from the database.
        :param table_name: name of the table to fetch
        :return: pandas DataFrame with the fetched table
        """
        query = f"SELECT * FROM {table_name}"
        return self.fetch_data(query)

# Ensure the main logic is outside of the class definition
if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        ingestion.fetch_and_store_data()
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise e

'''import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text

class PostgreSQLIngestion:
    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = None

    def connect(self):
        try:
            self.engine = create_engine(f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}")
            print("Database connection successful.")
        except Exception as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            raise

    def upsert_data(self, data, table_name):
        try:
            data.to_sql(table_name, self.engine, index=False, if_exists='replace')
            print(f"Data upserted into {table_name} successfully.")
        except Exception as e:
            print(f"Error upserting data into table {table_name}: {e}")

    def fetch_data(self, table_name):
        try:
            query = f"SELECT * FROM {table_name}"
            return pd.read_sql_query(query, self.engine)
        except Exception as e:
            print(f"Error fetching data from table {table_name}: {e}")
            return None

    def get_table_names(self):
        try:
            query = text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            with self.engine.connect() as connection:
                result = connection.execute(query)
                return [row[0] for row in result]  # Access the first item in each tuple
        except Exception as e:
            print(f"Error fetching table names: {e}")
            return []

    def close(self):
        if self.engine:
            self.engine.dispose()
        print("Database connection closed.")
'''