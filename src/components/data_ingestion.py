import pandas as pd
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
