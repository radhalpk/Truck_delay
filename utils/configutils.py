import os
import configparser
from sqlalchemy import create_engine, exc
import pandas as pd
import requests


CONFIG_FILE_PATH = 'C:\Desktop\Truck Project\src\config\config.ini'

# Function to read database connection parameters from a config file
def read_config(CONFIG_FILE_PATH):
    config = configparser.ConfigParser()
    config.read()
    
    db_config = {
        'username': config.get('DATABASE', 'username'),
        'password': config.get('DATABASE', 'password'),
        'host': config.get('DATABASE', 'host'),
        'port': config.get('DATABASE', 'port'),
        'database': config.get('DATABASE', 'dbname'),
        'github_url': config.get('API', 'github_url')

    }
    return db_config

# Function to create a database connection using SQLAlchemy
def get_connection(db_config):
    try:
        connection_string = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"Error occurred while creating database connection: {e}")
        return None

# Function to get all table names from the database
def get_table_names(engine):
    query = """
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public'
    """
    try:
        table_names = pd.read_sql(query, engine)
        return table_names['table_name'].tolist()
    except Exception as e:
        print(f"Error while fetching table names: {e}")
        return []

# Function to load all tables into separate pandas DataFrames
def load_all_tables(db_config):
    """
    Load all tables into DataFrames given the database connection config.
    """
    try:
        # Create a database connection
        db_url = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(db_url)
        
        # Get the table names
        with engine.connect() as connection:
            table_names = connection.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'").fetchall()
        
        # Read each table into a DataFrame and store it in a dictionary
        df_dict = {}
        for table_name in table_names:
            df_dict[table_name[0]] = pd.read_sql_table(table_name[0], engine)
        
        return df_dict

    except exc.SQLAlchemyError as e:
        print(f"Database error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None