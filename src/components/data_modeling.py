'''import hopsworks
import joblib
import mlflow
import hsml
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer



class ModelComponent:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.preprocessor = None

    def load_data_from_hopsworks(self):
        # Initialize Hopsworks connection
        project = hopsworks.login(api_key_value=self.config['api_key'])
        feature_store = project.get_feature_store()

        # Load dataset from Hopsworks feature store
        fg = feature_store.get_feature_group("final_df_feature_group", version=1)
        df = fg.read()

        # Check for null values and impute if necessary
        if df.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            df[df.columns] = imputer.fit_transform(df)

        # Handle outliers (using IQR method) for numeric columns
        for col in self.config['numerical_cols']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            df.loc[outliers, col] = df[col].median()

        return df

    def preprocess_data(self, df):
        # Convert 'estimated_arrival' to timezone-naive (remove timezone info)
        df['estimated_arrival'] = df['estimated_arrival'].dt.tz_localize(None)

        # Train-validation-test split based on 'estimated_arrival' date
        train_df = df[df['estimated_arrival'] <= pd.to_datetime('2019-01-30')]
        valid_df = df[(df['estimated_arrival'] > pd.to_datetime('2019-01-30')) &
                      (df['estimated_arrival'] <= pd.to_datetime('2019-02-07'))]
        test_df = df[df['estimated_arrival'] > pd.to_datetime('2019-02-07')]

        # Splitting features and target
        X_train = train_df[self.config['numerical_cols'] + self.config['categorical_cols']]
        y_train = train_df[self.config['target_column']]

        X_valid = valid_df[self.config['numerical_cols'] + self.config['categorical_cols']]
        y_valid = valid_df[self.config['target_column']]

        X_test = test_df[self.config['numerical_cols'] + self.config['categorical_cols']]
        y_test = test_df[self.config['target_column']]

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)

        # Preprocessing pipeline: OneHotEncode categorical variables, scale numeric variables, and apply PCA
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.config['numerical_cols']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.config['categorical_cols'])
            ]
        )

        # Apply the preprocessing pipeline
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_valid_preprocessed = preprocessor.transform(X_valid)
        X_test_preprocessed = preprocessor.transform(X_test)

        # Apply SMOTE after encoding
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=0.95)  # Retain 95% variance
        X_train_pca = pca.fit_transform(X_train_resampled)
        X_valid_pca = pca.transform(X_valid_preprocessed)
        X_test_pca = pca.transform(X_test_preprocessed)

        # Save the preprocessor and PCA for future use (e.g., during deployment)
        joblib.dump(preprocessor, 'preprocessor.pkl')
        joblib.dump(pca, 'pca_model.pkl')

        return X_train_pca, X_valid_pca, X_test_pca, y_train_resampled, y_valid, y_test

    def train_model(self, X_train, y_train, model_type='random_forest'):
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            }
            model = RandomForestClassifier(random_state=42)

        elif model_type == 'logistic_regression':
            param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2']}
            model = LogisticRegression(random_state=42)

        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6, 10],
                'colsample_bytree': [0.8, 1.0],
                'subsample': [0.8, 1.0],
                'gamma': [0, 0.1]
            }
            model = XGBClassifier(random_state=42)

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        joblib.dump(grid_search.best_estimator_, f'best_{model_type}_model.pkl')
        self.model = grid_search.best_estimator_

        return self.model, grid_search.best_params_

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }

        return metrics

    def log_model(self, model_type, local_model_dir, best_model, best_params, metrics):
        # Log to MLFlow
        with mlflow.start_run():
            mlflow.sklearn.log_model(best_model, "Best_Model")
            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("f1_score", metrics['f1_score'])
            mlflow.log_metric("precision", metrics['precision'])
            mlflow.log_metric("recall", metrics['recall'])

        # Create HSML connection for model registry
        connection = hsml.connection(api_key_value=self.config['api_key'])
        mr = connection.get_model_registry()

        # Save the model locally and register it in Hopsworks Model Registry
        os.makedirs(local_model_dir, exist_ok=True)
        mlflow.sklearn.save_model(best_model, path=local_model_dir)
        
        model = mr.sklearn.create_model(
            name=f"Best_{model_type}_Model",
            version=1,
            metrics={"accuracy": metrics['accuracy']},
            description=f"Best {model_type} Model with tuned hyperparameters."
        )
        model.save(local_model_dir)
        print(f"Model {model_type} successfully pushed to Hopsworks Model Registry!")

'''
import os
import pickle
import pandas as pd
import configparser
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from datetime import datetime, timezone
# Define the path to the configuration file
CONFIG_FILE_PATH = '/Users/pavankumarradhala/Desktop/projects/Truck_delay/src/config/config.ini'

class ModelTrainer:
    def __init__(self):
        config = configparser.RawConfigParser()
        config.read(CONFIG_FILE_PATH)
        # Add this debug statement
        print(f"Config Sections: {config.sections()}")
        self.model_dir = config.get('DATA', 'model_dir')

    def read_data(self, feature_store, feature_group_name):
        """Reads data from Hopsworks feature store."""
        try:
            fg_metadata = feature_store.get_feature_group(feature_group_name, version=1)
            if fg_metadata is None:
                raise ValueError(f"Feature group {feature_group_name} not found.")
            fg_df = fg_metadata.read() if not isinstance(fg_metadata, pd.DataFrame) else fg_metadata
            return fg_df
        except Exception as e:
            print(f"Error reading feature group {feature_group_name}: {e}")
            return None

    def split_data(self, final_merge):
        """Splits data into train, validation, and test sets based on date."""
        # Fixing the deprecation warning here by using `datetime.now(timezone.utc)`
        date_split_train = datetime.now(timezone.utc).replace(year=2019, month=1, day=30, hour=0, minute=0, second=0, microsecond=0)
        date_split_valid = datetime.now(timezone.utc).replace(year=2019, month=2, day=7, hour=0, minute=0, second=0, microsecond=0)


        train_df = final_merge[final_merge['estimated_arrival'] <= date_split_train]
        validation_df = final_merge[(final_merge['estimated_arrival'] > date_split_train) & 
                                    (final_merge['estimated_arrival'] <= date_split_valid)]
        test_df = final_merge[final_merge['estimated_arrival'] > date_split_valid]
        
        return train_df, validation_df, test_df

    '''def prepare_data(self, train_df, validation_df, test_df, cts_cols, categorical_cols, encode_columns):
        """Prepares data for training by encoding specified categorical features and scaling continuous features."""
        
        # Separate target variable (delay)
        y_train = train_df['delay']
        y_valid = validation_df['delay']
        y_test = test_df['delay']
        
        # Feature sets (X)
        X_train = train_df[cts_cols + categorical_cols].copy()
        X_valid = validation_df[cts_cols + categorical_cols].copy()
        X_test = test_df[cts_cols + categorical_cols].copy()
        
        # Reset the index to avoid misalignment after transformations
        X_train.reset_index(drop=True, inplace=True)
        X_valid.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_valid.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        # Convert all integer columns to float64
        X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes('int').columns})
        X_valid = X_valid.astype({col: 'float64' for col in X_valid.select_dtypes('int').columns})
        X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes('int').columns})

        # Encoding categorical columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_encoded = encoder.fit_transform(X_train[encode_columns])
        X_valid_encoded = encoder.transform(X_valid[encode_columns])
        X_test_encoded = encoder.transform(X_test[encode_columns])
        
        # Save the encoder in the model directory
        encoder_file_path = os.path.join(self.model_dir, 'encoder.pkl')
        with open(encoder_file_path, 'wb') as f:
           pickle.dump(encoder, f)
        # Convert the encoded arrays back into DataFrames with column names
        encoded_train_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(encode_columns))
        encoded_valid_df = pd.DataFrame(X_valid_encoded, columns=encoder.get_feature_names_out(encode_columns))
        encoded_test_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(encode_columns))

        # Concatenate the original features (cts_cols) and encoded categorical features all at once
        X_train = pd.concat([X_train.drop(columns=encode_columns), encoded_train_df], axis=1)
        X_valid = pd.concat([X_valid.drop(columns=encode_columns), encoded_valid_df], axis=1)
        X_test = pd.concat([X_test.drop(columns=encode_columns), encoded_test_df], axis=1)
        
        # After concatenation, make sure target variables are aligned with features
        y_train = y_train.loc[X_train.index]
        y_valid = y_valid.loc[X_valid.index]
        y_test = y_test.loc[X_test.index]
        
        # Scaling continuous columns
        scaler = StandardScaler()
        X_train[cts_cols] = scaler.fit_transform(X_train[cts_cols])
        X_valid[cts_cols] = scaler.transform(X_valid[cts_cols])
        X_test[cts_cols] = scaler.transform(X_test[cts_cols])
        
        print(X_train.isnull().sum().sum())
        print(X_test.isnull().sum().sum())
        print(X_valid.isnull().sum().sum())
        
        # Save the scaler in the model directory
        scaler_file_path = os.path.join(self.model_dir, 'standardscaler.pkl')
        with open(scaler_file_path, 'wb') as f:
           pickle.dump(scaler, f)
        return X_train, X_valid, X_test, y_train, y_valid, y_test'''
    
    def prepare_data(self, train_df, validation_df, test_df, cts_cols, categorical_cols, encode_columns):
       """Prepares data for training by encoding specified categorical features and scaling continuous features."""
    
    # Separate target variable (delay)
       y_train = train_df['delay']
       y_valid = validation_df['delay']
       y_test = test_df['delay']
    
    # Feature sets (X)
       X_train = train_df[cts_cols + categorical_cols].copy()
       X_valid = validation_df[cts_cols + categorical_cols].copy()
       X_test = test_df[cts_cols + categorical_cols].copy()
    
    # Reset the index to avoid misalignment after transformations
       X_train.reset_index(drop=True, inplace=True)
       X_valid.reset_index(drop=True, inplace=True)
       X_test.reset_index(drop=True, inplace=True)
       y_train.reset_index(drop=True, inplace=True)
       y_valid.reset_index(drop=True, inplace=True)
       y_test.reset_index(drop=True, inplace=True)

    # Convert all integer columns to float64
       X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes('int').columns})
       X_valid = X_valid.astype({col: 'float64' for col in X_valid.select_dtypes('int').columns})
       X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes('int').columns})

    # Encoding categorical columns
       encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
       X_train_encoded = encoder.fit_transform(X_train[encode_columns])
       X_valid_encoded = encoder.transform(X_valid[encode_columns])
       X_test_encoded = encoder.transform(X_test[encode_columns])

    # Ensure model directory exists before saving
       if not os.path.exists(self.model_dir):
           os.makedirs(self.model_dir)

    # Save the encoder in the model directory
       encoder_file_path = os.path.join(self.model_dir, 'encoder.pkl')
       try:
           with open(encoder_file_path, 'wb') as f:
               pickle.dump(encoder, f)
       except Exception as e:
           print(f"Error saving encoder: {e}")
    
    # Convert the encoded arrays back into DataFrames with column names
       encoded_train_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(encode_columns))
       encoded_valid_df = pd.DataFrame(X_valid_encoded, columns=encoder.get_feature_names_out(encode_columns))
       encoded_test_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(encode_columns))

    # Concatenate the original features (cts_cols) and encoded categorical features all at once
       X_train = pd.concat([X_train.drop(columns=encode_columns), encoded_train_df], axis=1)
       X_valid = pd.concat([X_valid.drop(columns=encode_columns), encoded_valid_df], axis=1)
       X_test = pd.concat([X_test.drop(columns=encode_columns), encoded_test_df], axis=1)
    
    # After concatenation, make sure target variables are aligned with features
       y_train = y_train.loc[X_train.index]
       y_valid = y_valid.loc[X_valid.index]
       y_test = y_test.loc[X_test.index]
    
    # Scaling continuous columns
       scaler = StandardScaler()
       X_train[cts_cols] = scaler.fit_transform(X_train[cts_cols])
       X_valid[cts_cols] = scaler.transform(X_valid[cts_cols])
       X_test[cts_cols] = scaler.transform(X_test[cts_cols])
    
    # Print any null values
       print(X_train.isnull().sum().sum())
       print(X_test.isnull().sum().sum())
       print(X_valid.isnull().sum().sum())
    
    # Save the scaler in the model directory
       scaler_file_path = os.path.join(self.model_dir, 'standardscaler.pkl')
       try:
           with open(scaler_file_path, 'wb') as f:
               pickle.dump(scaler, f)
       except Exception as e:
           print(f"Error saving scaler: {e}")

       return X_train, X_valid, X_test, y_train, y_valid, y_test

    def find_best_model(self, X_train, y_train, X_valid, y_valid, model_params):
        """Performs hyperparameter tuning for Logistic Regression, RandomForest, and XGBoost using GridSearch."""
        
        best_model = None
        best_score = 0
        best_model_name = None
        best_model_params = None

        for model_name, model_data in model_params.items():
            model = model_data['model']
            params = model_data['parameters']
            
            print(f"Training model: {model_name}...")
            
            # Perform grid search
            grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=10, verbose=0, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Get the best estimator from grid search
            best_model_candidate = grid_search.best_estimator_
            
            # Create a nested run to log parameters without overwriting
            with mlflow.start_run(nested=True):
                try:
                    # Log best parameters after fitting
                    for param_key, param_value in grid_search.best_params_.items():
                        mlflow.log_param(param_key, param_value)
                except mlflow.MlflowException as e:
                    print(f"Error logging parameters for {model_name}: {e}")
            
            # Calculate validation accuracy
            score = accuracy_score(y_valid, best_model_candidate.predict(X_valid))
            
            print(f"{model_name} validation accuracy: {score}")
            mlflow.log_metric(f'{model_name}_validation_score', score)
            print(f"Best params for {model_name}: {grid_search.best_params_}")
            print(f"Validation score for {model_name}: {score}")

            # Update best model based on validation score
            if score > best_score:
                best_score = score  # Use the validation score for comparison
                best_model_name = model_name
                best_model = best_model_candidate
                best_model_params = grid_search.best_params_

        print(f"\nBest Model: {best_model_name} with validation accuracy: {best_score}")
        return best_model_name, best_model, best_model_params

    def final_model(self, X_train, X_test, y_train, y_test, best_model, best_model_params):
        """Trains the final model using the best parameters and evaluates it on the test set."""
        print("Training the best model on full train data...")

        best_model.fit(X_train, y_train)
        y_pred_test = best_model.predict(X_test)
        y_pred_train=best_model.predict(X_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        report_test = classification_report(y_test, y_pred_test)
        report_train= classification_report(y_train,y_pred_train)
        
        print(f"Train Accuracy: {accuracy_train}")
        print(f"Test Accuracy: {accuracy_test}")
        print(f"Classification Report for Test:\n{report_test}")
        print(f"Classification Report for Train:\n{report_train}")

        mlflow.log_metric("train_accuracy",accuracy_train)
        mlflow.log_metric("test_accuracy", accuracy_test)
        
        # Calculate F1 score
        f1_train = f1_score(y_train, y_pred_train, average='weighted')
        f1_test = f1_score(y_test, y_pred_test, average='weighted')
        
        # Log the classification report as artifacts
        with open("classification_report_test.txt", "w") as f_test:
            f_test.write(report_test)
        with open("classification_report_train.txt", "w") as f_train:
            f_train.write(report_train)
        
        mlflow.log_artifact("classification_report_test.txt")
        mlflow.log_artifact("classification_report_train.txt")
        mlflow.log_metric("train_f1_score", f1_train)
        mlflow.log_metric("test_f1_score", f1_test)
        
        '''# Save the model using pickle
        model_file_path = os.path.join(self.model_dir, 'best_model.pkl')
        with open(model_file_path, 'wb') as f:
            pickle.dump(best_model, f)
        input_example = X_train.sample(5)'''
        # Save the best model using pickle
        model_file_path = os.path.join(self.model_dir, 'best_model.pkl')
        try:
            with open(model_file_path, 'wb') as f:
                  pickle.dump(best_model, f)
        except Exception as e:
            print(f"Error saving model: {e}")

        # Infer the signature (input and output types)
        input_example = X_train.sample(5)
        signature = infer_signature(X_train, best_model.predict(X_train))

        # Log the model with signature and input example
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="best_model", 
            input_example=input_example, 
            signature=signature
        )
        
        # Log the encoder and scaler as artifacts in MLflow from the model directory
        encoder_file_path = os.path.join(self.model_dir, 'encoder.pkl')
        scaler_file_path = os.path.join(self.model_dir, 'standardscaler.pkl')
        mlflow.log_artifact(encoder_file_path)
        mlflow.log_artifact(scaler_file_path)
        # Create a dictionary of metrics to pass to Hopsworks
        metrics = {
        "train_accuracy": accuracy_train,
        "test_accuracy": accuracy_test,
        "train_f1_score": f1_train,
        "test_f1_score": f1_test
        }
        
        return {
        "model_file_path": model_file_path,
        "metrics": metrics
        }
    
    def register_model_in_hopsworks(self, project, model_name, metrics, model_file_path):
        """Registers the trained model in the Hopsworks Model Registry and prepares it for deployment."""
        try:
            # Get the model registry handle for the project's model registry
            mr = project.get_model_registry()

            # Check if the model already exists
            try:
                existing_model = mr.get_model(name=model_name, version=1)
                print(f"Model {model_name} version 1 already exists. Deleting the old model...")
                existing_model.delete()  # Delete the existing model version 1
            except Exception as e:
                print(f"No existing model found with name {model_name} version 1.")
            
            # Create a new model in the Hopsworks Model Registry
            hopsworks_model = mr.python.create_model(
                name=model_name,
                version=1,  # you can dynamically increase version
                metrics=metrics,  # dictionary of your model's performance metrics (accuracy, f1, etc.)
                description=f"{model_name} model for predicting truck delays"
            )

            # Upload the model to the model registry
            hopsworks_model.save(model_file_path)  # Save the model in Hopsworks Model Registry

            print(f"Model {model_name} registered successfully in Hopsworks.")

        except Exception as e:
            print(f"Error registering model in Hopsworks: {e}")


# Model Serving and Deployment in Hopsworks:
# python
# Copy code
# def deploy_model_in_hopsworks(self, project, model_name):
#     """Deploys the best performing model in Hopsworks for serving."""
#     try:
#         # Get the model registry handle
#         mr = project.get_model_registry()

#         # Get the best performing model by accuracy
#         best_model = mr.get_best_model(model_name, 'accuracy', 'max')

#         # Deploy the best model
#         deployment = best_model.deploy()

#         print(f"Deployment started for model {model_name}.")

#         # Start the deployment
#         deployment.start()
#         print(f"Model {model_name} deployed successfully.")
#     except Exception as e:
#         print(f"Error during model deployment: {e}")
