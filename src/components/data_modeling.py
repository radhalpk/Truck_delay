import hopsworks
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


'''import hopsworks
import joblib
import mlflow
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
        project = hopsworks.login()
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

    def log_model(self, model_type):
        mlflow.sklearn.log_model(self.model, f"Best_{model_type}_Model")
        mlflow.log_params(self.config)

        project = hopsworks.login()
        model_registry = project.get_model_registry()
        model_registry.sklearn.create_model(self.model, model_name=f"Best_{model_type}_Model")'''
        
'''import hsml
import joblib
import mlflow
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
        project = hopsworks.login()
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

    def log_model(self, model_type, best_model, best_params, test_accuracy, test_f1, test_precision, test_recall):
        # Set up HSML connection
        connection = hsml.connection()  # Not passing project or host since your notebook worked without it.
        
        # Get the model registry handle
        mr = connection.get_model_registry()
        
        # Start an MLflow run
        with mlflow.start_run():
            # Log the best model using MLFlow
            mlflow.sklearn.log_model(best_model, f"Best_{model_type}_Model")

            # Log the hyperparameters and metrics
            mlflow.log_params(best_params)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_f1_score", test_f1)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)

        # Save the model locally to the desired directory
        local_model_dir = "/Users/pavankumarradhala/Desktop/projects/Truck_delay/model"
        mlflow.sklearn.save_model(best_model, path=local_model_dir)

        # Register the model in Hopsworks Model Registry
        model = mr.sklearn.create_model(
            name=f"Best_{model_type}_Model",
            version=1,
            metrics={"accuracy": test_accuracy},
            description=f"Best {model_type} Model with tuned hyperparameters."
        )
        
        # Save the model in the registry
        model.save(local_model_dir)

        print("Model successfully pushed to Hopsworks Model Registry!")'''
