import sys
import os
import os.path as path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)



# Dynamic path to the config file
#CONFIG_FILE_PATH = os.path.join(project_root, 'src', 'config', 'config.ini')

from pipelines.stage_01_data_ingestion import DataIngestion
from pipelines.stage_02_data_cleaning import DataCleaningPipeline
from pipelines.stage_03_data_Transformation import FeatureEngineeringPipeline
from pipelines.Model_training import TruckDelayModelingPipeline

# Path to the config file
CONFIG_FILE_PATH = '/Users/pavankumarradhala/Desktop/projects/Truck_delay/src/config/config.ini'
#CONFIG_FILE_PATH = os.path.join(project_root, 'src', 'config', 'config.ini')

STAGE_NAME = "Data Ingestion"
try:
   print(">>>>>> Stage {STAGE_NAME} started <<<<<<") 
   ingestion = DataIngestion(CONFIG_FILE_PATH)
   # Fetch and store data
   ingestion.ingest_data()

        # Close the database connection after all operations
   ingestion.close_database_connection()
   print(">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print((e))
        raise e
     
STAGE_NAME = "DATA CLEANING"
try:
   print(">>>>>> Stage {STAGE_NAME} started <<<<<<",STAGE_NAME) 
   data_cleaning = DataCleaningPipeline()
   data_cleaning.main()
   print(">>>>>> Stage {STAGE_NAME} completed <<<<<<",STAGE_NAME)
except Exception as e:
        print(e)
        raise e
          
STAGE_NAME = "Data Transformation"
try:
   print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
   obj = FeatureEngineeringPipeline()
   obj.main()
   print(">>>>>> Stage completed :", STAGE_NAME)
except Exception as e:
   print(e)
   raise e

STAGE_NAME = "Model Training"
try:
   print(">>>>>> Stage started <<<<<<:",STAGE_NAME)
   obj = TruckDelayModelingPipeline()
   obj.main()
   print(">>>>>> Stage  completed <<<<<<>:", STAGE_NAME)
except Exception as e:
   print(e)
   raise e
'''import sys
import os
import os.path as path

# Define project root dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Dynamic path to the config file
CONFIG_FILE_PATH = os.path.join(project_root, 'src', 'config', 'config.ini')

from pipelines.stage_01_data_ingestion import DataIngestion
from pipelines.stage_02_data_cleaning import DataCleaningPipeline
from pipelines.stage_03_data_Transformation import FeatureEngineeringPipeline
from pipelines.Model_training import TruckDelayModelingPipeline

# Stages

STAGE_NAME = "Data Ingestion"
try:
    print(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    ingestion = DataIngestion(CONFIG_FILE_PATH)
    # Fetch and store data
    ingestion.ingest_data()

    # Close the database connection after all operations
    ingestion.close_database_connection()
    print(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    print(e)
    raise e

STAGE_NAME = "DATA CLEANING"
try:
    print(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_cleaning = DataCleaningPipeline()
    data_cleaning.main()
    print(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    print(e)
    raise e

STAGE_NAME = "Data Transformation"
try:
    print(f">>>>>> Stage started <<<<<< : {STAGE_NAME}")
    obj = FeatureEngineeringPipeline()
    obj.main()
    print(f">>>>>> Stage completed : {STAGE_NAME}")
except Exception as e:
    print(e)
    raise e

STAGE_NAME = "Model Training"
try:
    print(f">>>>>> Stage started <<<<<<: {STAGE_NAME}")
    obj = TruckDelayModelingPipeline()
    obj.main()
    print(f">>>>>> Stage completed <<<<<<: {STAGE_NAME}")
except Exception as e:
    print(e)
    raise e'''
