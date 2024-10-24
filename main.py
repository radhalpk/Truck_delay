import sys
import os
import os.path as path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from pipelines.stage_01_data_ingestion import DataIngestion
from pipelines.stage_02_data_cleaning import DataCleaningPipeline
from pipelines.data_modeling_pipeline import TruckDelayModelingPipeline
from pipelines.data_merging_pipeline import FeatureEngineeringPipeline

STAGE_NAME = "Data Ingestion"

try:
   print(">>>>>> Stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestion()
   data_ingestion.main()
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
   obj = TruckDelayModelingPipeline()
   obj.main()
   print(">>>>>> Stage completed :", STAGE_NAME)
except Exception as e:
   print(e)
   raise e

STAGE_NAME = "Model Training"
try:
   print(">>>>>> Stage started <<<<<<:",STAGE_NAME)
   obj = FeatureEngineeringPipeline()
   obj.main()
   print(">>>>>> Stage  completed <<<<<<>:", STAGE_NAME)
except Exception as e:
   print(e)
   raise e