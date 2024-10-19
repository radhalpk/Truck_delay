'''import sys
import os
import configparser


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
    # Instantiate the ModelComponent class
    model_component = ModelComponent(config)

    # Step 1: Load data from Hopsworks
    print("Loading data from Hopsworks...")
    df = model_component.load_data_from_hopsworks()

    # Step 2: Preprocess data
    print("Preprocessing data...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = model_component.preprocess_data(df)

    # Step 3: Train the model
    print(f"Training the {model_type} model...")
    best_model, best_params = model_component.train_model(X_train, y_train, model_type=model_type)
    print(f"Best {model_type} Model Parameters: {best_params}")

    # Step 4: Evaluate the model
    print(f"Evaluating the {model_type} model...")
    metrics = model_component.evaluate_model(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")

    # Step 5: Log model to MLFlow and Hopsworks Model Registry
    model_component.log_model(model_type)


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
import sys
import os
import configparser
import mlflow
import mlflow.sklearn
import hsml  # Added for Hopsworks Model Registry connection


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
    # Step 1: Set the MLflow Tracking URI for Hopsworks
    mlflow.set_tracking_uri("https://c.app.hopsworks.ai:443/your_project_id/mlflow")  # Replace 'your_project_id' with your actual project ID

    # Instantiate the ModelComponent class
    model_component = ModelComponent(config)

    # Step 2: Load data from Hopsworks
    print("Loading data from Hopsworks...")
    df = model_component.load_data_from_hopsworks()

    # Step 3: Preprocess data
    print("Preprocessing data...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = model_component.preprocess_data(df)

    # Step 4: Start an MLflow run to track the experiment
    with mlflow.start_run():
        # Step 5: Train the model
        print(f"Training the {model_type} model...")
        best_model, best_params = model_component.train_model(X_train, y_train, model_type=model_type)
        print(f"Best {model_type} Model Parameters: {best_params}")

        # Step 6: Evaluate the model
        print(f"Evaluating the {model_type} model...")
        metrics = model_component.evaluate_model(X_test, y_test)
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"F1 Score: {metrics['f1_score']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")

        # Step 7: Log model, parameters, and metrics to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        # Log the model to MLflow and Hopsworks Model Registry
        model_component.log_model(model_type)

        # Step 8: Connect to Hopsworks Model Registry
        try:
            api_key = config_parser['API']['hopswork_api_key']  # Load API key from config
            connection = hsml.connection(
                host="c.app.hopsworks.ai", 
                port=443, 
                project="1044630",  # Replace with your project ID
                api_key_value=api_key
            )
            connection.connect()

            # Get the model registry handle for the project's model registry
            mr = connection.get_model_registry()

            # Create a new model in the registry
            model = mr.tensorflow.create_model(
                name="truck_delay_model", 
                version=1, 
                metrics={"accuracy": metrics['accuracy']}, 
                description="Truck delay prediction model"
            )

            # Save the model to Hopsworks Model Registry
            model.save("/your/local/model_directory")  # Update with your model directory path
            print("Model successfully saved to Hopsworks Model Registry.")

        except Exception as e:
            print(f"Error in connecting to Hopsworks Model Registry: {e}")

        # End the MLflow run
        mlflow.end_run()


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
        raise e
