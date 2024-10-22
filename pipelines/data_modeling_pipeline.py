'''import sys
import os
import configparser
import mlflow
import mlflow.sklearn
import hsml
import hopsworks

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.components.data_modeling import ModelComponent

# Initialize configparser and read config
config_parser = configparser.ConfigParser()
config_parser.read('/Users/pavankumarradhala/Desktop/projects/Truck_delay/src/config/config.ini')

# Define the configuration
config = {
    'api_key': config_parser['API']['hopswork_api_key'],
    'numerical_cols': ['route_avg_temp', 'route_avg_wind_speed', 'route_avg_precip', 'route_avg_humidity', 
                       'route_avg_visibility', 'route_avg_pressure', 'distance', 'average_hours',
                       'temp_origin', 'wind_speed_origin', 'precip_origin', 'humidity_origin', 
                       'visibility_origin', 'pressure_origin', 'temp_destination', 'wind_speed_destination',
                       'precip_destination', 'humidity_destination', 'visibility_destination',
                       'pressure_destination', 'avg_no_of_vehicles', 'truck_age', 'load_capacity_pounds', 
                       'mileage_mpg', 'age', 'experience', 'average_speed_mph'],
    'categorical_cols': ['route_description', 'description_origin', 'description_destination', 'accident', 
                         'fuel_type', 'gender', 'driving_style', 'ratings', 'is_midnight'],
    'target_column': 'delay'
}

def run_model_pipeline(model_type='random_forest'):
    # Set the MLflow Tracking URI for Hopsworks
    mlflow.set_tracking_uri("https://c.app.hopsworks.ai:443/1044630/mlflow")  # Replace with your project ID

    # Instantiate the ModelComponent class
    model_component = ModelComponent(config)

    # Load data from Hopsworks
    print("Loading data from Hopsworks...")
    df = model_component.load_data_from_hopsworks()

    # Preprocess data
    print("Preprocessing data...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = model_component.preprocess_data(df)

    # Train the model
    print(f"Training the {model_type} model...")
    best_model, best_params = model_component.train_model(X_train, y_train, model_type=model_type)
    print(f"Best {model_type} Model Parameters: {best_params}")

    # Evaluate the model
    print(f"Evaluating the {model_type} model...")
    metrics = model_component.evaluate_model(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")

    # Create an HSML connection (no project ID needed here, as we are logging in)
    connection = hsml.connection(api_key_value=config['api_key'])
    mr = connection.get_model_registry()

    # Start an MLflow run to log the model and metrics
    with mlflow.start_run():
        # Log the best model using MLFlow
        mlflow.sklearn.log_model(best_model, "Best_Model")

        # Log the hyperparameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])

    # Specify your custom directory where you want to save the model
    local_model_dir = "/Users/pavankumarradhala/Desktop/projects/Truck_delay/model"
    os.makedirs(local_model_dir, exist_ok=True)

    # Save the model locally to the specified directory
    mlflow.sklearn.save_model(best_model, path=local_model_dir)

    # Register the model in Hopsworks Model Registry
    model = mr.sklearn.create_model(
        name="Best_Model",
        version=1,
        metrics={"accuracy": metrics['accuracy']},
        description=f"Best {model_type} Model with tuned hyperparameters."
    )
    # Save the model in the registry
    model.save(local_model_dir)

    print("Model successfully pushed to Hopsworks Model Registry!")


# Run the pipeline
if __name__ == "__main__":
    try:
        STAGE_NAME = "DATA_MODELING"
        print(f">>>>>> Stage started <<<<<< : {STAGE_NAME}")

        # Run the model pipeline
        run_model_pipeline(model_type='random_forest')  # You can switch to 'logistic_regression' or 'xgboost'

        print(f">>>>>> Stage completed <<<<<< : {STAGE_NAME}")

    except Exception as e:
        print(f"Error in stage: {STAGE_NAME}")
        print(e)
        raise e'''



import os.path as path
import os
import sys
from IPython.display import display
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.components.data_modeling import ModelTrainer
from src.config import *
import configparser
import pandas as pd
import hopsworks
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
# Configuration file path
CONFIG_FILE_PATH = "/Users/pavankumarradhala/Desktop/projects/Truck_delay/src/config/config.ini"

# Configuration setup
config = configparser.RawConfigParser()
config.read(CONFIG_FILE_PATH)
MODEL_DIR = config.get('DATA', 'model_dir')

# Feature store setup
project = hopsworks.login()
fs = project.get_feature_store()

# Pipeline Class
class TruckDelayModelingPipeline:
    def __init__(self):
        config = configparser.RawConfigParser()
        config.read(CONFIG_FILE_PATH)

        # Debug: Verify if the file exists
        if not os.path.exists(CONFIG_FILE_PATH):
            print(f"Configuration file not found at: {CONFIG_FILE_PATH}")
        else:
            print(f"Configuration file found at: {CONFIG_FILE_PATH}")

        # Debug: Print the sections in the config file
        print("Config Sections Loaded:", config.sections())

        # Now attempt to get the 'model_dir'
        self.model_trainer = ModelTrainer()

    def main(self):
        try:
            mlflow.set_experiment('TruckDelayModeling')
            with mlflow.start_run():
                # Retrieve dataset from Hopsworks
                final_merge = self.model_trainer.read_data(fs, 'final_df_feature_group')
                if final_merge is None:
                    raise ValueError("Failed to load data from the feature store.")

                # Verify dataset structure
                self.verify_dataset(final_merge)

                # Preprocess the data
                final_merge = self.preprocess_data(final_merge)

                # Split the data into train, validation, and test sets
                train_df, validation_df, test_df = self.model_trainer.split_data(final_merge)

                # Define column names
                cts_cols = self.get_continuous_columns()
                cat_cols = self.get_categorical_columns()
                encode_columns = self.get_encoder_columns()

                # Prepare data (encode categorical and scale continuous features)
                X_train, X_valid, X_test, y_train, y_valid, y_test = self.model_trainer.prepare_data(
                    train_df, validation_df, test_df, cts_cols, cat_cols, encode_columns
                )

                # Hyperparameter tuning and model evaluation
                model_params = self.get_model_params()
                best_model_name, best_model, best_model_params = self.model_trainer.find_best_model(
                    X_train, y_train, X_valid, y_valid, model_params
                )

                # Log best model name and params in MLflow
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_params(best_model_params)

                # Final model evaluation
                if best_model_name and best_model:
                    print(f"Best Model: {best_model_name}")
                    print(f"Best Parameters: {best_model_params}")
                    result = self.model_trainer.final_model(
                        X_train, X_test, y_train, y_test, best_model, best_model_params
                    )
                    model_file_path = result["model_file_path"]
                    metrics = result["metrics"]
                    # Register the final model in Hopsworks Model Registry
                    self.model_trainer.register_model_in_hopsworks(project, best_model_name, metrics, model_file_path)

        except Exception as e:
            mlflow.log_artifact("Error during model training")
            print(f"Error: {e}")
            raise e

    def verify_dataset(self, final_merge):
        """Verifies the structure and columns of the dataset."""
        print(f"Dataset shape: {final_merge.shape}")
        print(f"Dataset columns: {final_merge.columns}")
        print(f"Data types:\n{final_merge.dtypes}")
        print(f"Null values:\n{final_merge.isnull().sum()}")

    def preprocess_data(self, final_merge):
        """Preprocesses the data by handling missing values and dropping irrelevant columns."""
        # Drop irrelevant columns
        final_merge = final_merge.drop([ 
            'unique_id', 'truck_id', 'route_id', 'origin_id', 'destination_id',
            'driver_id', 'departure_date', 'estimated_arrival_nearest_hour', 'departure_date_nearest_hour', 'name'
        ], axis=1)
        
        # Handle missing values - Drop rows with missing target variable 'delay'
        final_merge = final_merge.dropna(subset=['delay'])
        return final_merge

    def get_continuous_columns(self):
        """Returns a list of continuous columns."""
        return ['route_avg_temp', 'route_avg_wind_speed', 'route_avg_precip', 'route_avg_humidity', 
                'route_avg_visibility', 'route_avg_pressure', 'distance', 'average_hours', 'temp_origin', 
                'wind_speed_origin', 'precip_origin', 'humidity_origin', 'visibility_origin', 'pressure_origin',
                'temp_destination', 'wind_speed_destination', 'precip_destination', 'humidity_destination', 
                'visibility_destination', 'pressure_destination', 'avg_no_of_vehicles', 'truck_age', 
                'load_capacity_pounds', 'mileage_mpg', 'age', 'experience', 'average_speed_mph']

    def get_categorical_columns(self):
        """Returns a list of categorical columns."""
        return ['route_description', 'description_origin', 'description_destination', 'accident', 'fuel_type', 
                'gender', 'driving_style', 'ratings', 'is_midnight']
        
    def get_encoder_columns(self):
        return ['route_description', 'description_origin', 'description_destination', 'fuel_type', 'gender', 'driving_style']

    def get_model_params(self):
        """Returns hyperparameter grids for different models."""
        return {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=5000),
                'parameters': {
                    'C': [0.001, 0.01, 0.1],
                    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga'],
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'parameters': {
                    'n_estimators': [200,300,400],
                    'max_depth': [3,4,5],
                    'min_samples_split': [3,4, 5],
                    'min_samples_leaf': [2, 3, 4],
                }
            },
            'xgboost': {
                'model': XGBClassifier(),
                'parameters': {
                    'max_depth': [4,5],             # Control tree depth (reduce to avoid overfitting)
                    'n_estimators': [400,500],     # Control the number of trees
                    'learning_rate': [0.01],        # Lower learning rate can help with generalization
                    'subsample': [0.4, 0.5, 0.6],        # Fraction of samples to use per tree
                }
            }
        }

if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< : MODEL TRAINING")
        obj = TruckDelayModelingPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<< : MODEL TRAINING")
    except Exception as e:
        print(e)
        raise e