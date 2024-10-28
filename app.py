'''import zipfile
import streamlit as st
import hopsworks
import mlflow
import joblib
import pandas as pd
from mlflow.tracking import MlflowClient

# Initialize Hopsworks connection and get the dataset
project = hopsworks.login()
fs = project.get_feature_store()

# Load the final merge dataset from Hopsworks Feature Store
final_merge_fg = fs.get_feature_group("final_df_feature_group", version=1)
final_merge = final_merge_fg.read()

# Access the Hopsworks Model Registry and retrieve the latest version of the xgboost model
model_registry = project.get_model_registry()
model_entity = model_registry.get_model("xgboost", version=1)
model_dir = model_entity.download()

# Extract model artifacts
zip_file_path = f"{model_dir}/xgboost_artifacts.zip"
extracted_dir = f"{model_dir}/xgboost_artifacts"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Load the model, encoder, and scaler
model = joblib.load(f"{extracted_dir}/best_model.pkl")
encoder = joblib.load(f"{extracted_dir}/encoder.pkl")
scaler = joblib.load(f"{extracted_dir}/scaler.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Truck Delay Prediction", layout="wide")
st.title('🚚 Truck Delay Prediction')

# Sidebar for filter options
st.sidebar.header("Filter Options")
filter_type = st.sidebar.radio("Select Filter Type", ['By Date Range', 'By Truck ID', 'By Route ID'])

if filter_type == 'By Date Range':
    st.sidebar.subheader("Date Range")
    from_date = st.sidebar.date_input("Start Date", value=min(final_merge['departure_date']))
    to_date = st.sidebar.date_input("End Date", value=max(final_merge['departure_date']))

elif filter_type == 'By Truck ID':
    st.sidebar.subheader("Truck ID")
    truck_id = st.sidebar.selectbox('Select Truck ID', final_merge['truck_id'].unique())

elif filter_type == 'By Route ID':
    st.sidebar.subheader("Route ID")
    route_id = st.sidebar.selectbox('Select Route ID', final_merge['route_id'].unique())

# Main section: Prediction and results
st.markdown("### Prediction Results")

# Button to trigger prediction
if st.sidebar.button("Predict"):
    try:
        # Apply the appropriate filter based on the user's selection
        if filter_type == 'By Date Range':
            filtered_data = final_merge[(final_merge['departure_date'] >= str(from_date)) & 
                                        (final_merge['departure_date'] <= str(to_date))].copy()
        elif filter_type == 'By Truck ID':
            filtered_data = final_merge[final_merge['truck_id'] == truck_id].copy()
        elif filter_type == 'By Route ID':
            filtered_data = final_merge[final_merge['route_id'] == route_id].copy()

        # Drop columns not needed for prediction
        if 'delay' in filtered_data.columns:
            filtered_data = filtered_data.drop(columns=['delay', 'unique_id'])

        # Preprocess the data: select continuous, categorical, and object columns
        cts_cols = ['route_avg_temp', 'route_avg_wind_speed', 'route_avg_precip', 'route_avg_humidity', 
                    'route_avg_visibility', 'route_avg_pressure', 'distance', 'average_hours', 'temp_origin', 
                    'wind_speed_origin', 'precip_origin', 'humidity_origin', 'visibility_origin', 'pressure_origin',
                    'temp_destination', 'wind_speed_destination', 'precip_destination', 'humidity_destination', 
                    'visibility_destination', 'pressure_destination', 'avg_no_of_vehicles', 'truck_age', 
                    'load_capacity_pounds', 'mileage_mpg', 'age', 'experience', 'average_speed_mph']
        
        encode_columns = ['route_description', 'description_origin', 'description_destination', 'fuel_type', 'gender', 'driving_style']
        object_cols = ['accident', 'ratings', 'is_midnight']
        
        # Convert object columns to numeric
        filtered_data[object_cols] = filtered_data[object_cols].apply(pd.to_numeric, errors='coerce')

        # Prepare feature data (X)
        X_filtered = filtered_data[cts_cols + object_cols + encode_columns].copy()
        X_filtered.reset_index(drop=True, inplace=True)

        # Apply encoding and scaling
        X_filtered_encoded = encoder.transform(X_filtered[encode_columns])
        encoded_filtered_df = pd.DataFrame(X_filtered_encoded, columns=encoder.get_feature_names_out(encode_columns))
        X_filtered_final = pd.concat([X_filtered.drop(columns=encode_columns).reset_index(drop=True), encoded_filtered_df], axis=1)
        X_filtered_final[cts_cols] = scaler.transform(X_filtered_final[cts_cols])

        # Make predictions using the loaded model
        predictions = model.predict(X_filtered_final)
        filtered_data['Predicted Delay'] = predictions

        # Display the results
        st.success(f"Predictions complete! Found {len(filtered_data)} records.")
        st.dataframe(filtered_data)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Please apply filters and click 'Predict' to see results.")'''
    
    
import zipfile
import streamlit as st
import hopsworks
import mlflow
import joblib
import pandas as pd
from mlflow.tracking import MlflowClient

# Initialize Hopsworks connection and get the dataset
project = hopsworks.login()
fs = project.get_feature_store()

# Load the final merge dataset from Hopsworks Feature Store
final_merge_fg = fs.get_feature_group("final_df_feature_group", version=1)
final_merge = final_merge_fg.read()

# Access the Hopsworks Model Registry and retrieve the latest version of the xgboost model
model_registry = project.get_model_registry()
model_entity = model_registry.get_model("xgboost", version=1)
model_dir = model_entity.download()

# Extract model artifacts
zip_file_path = f"{model_dir}/xgboost_artifacts.zip"
extracted_dir = f"{model_dir}/xgboost_artifacts"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Load the model, encoder, and scaler
model = joblib.load(f"{extracted_dir}/best_model.pkl")
encoder = joblib.load(f"{extracted_dir}/encoder.pkl")
scaler = joblib.load(f"{extracted_dir}/scaler.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Truck Delay Prediction", layout="wide")
st.title('🚛 Truck Delay Prediction')

# Add description
st.markdown("""
Welcome to the Truck Delay Prediction App! 📊

Use this app to predict potential delays in truck schedules based on various factors like weather conditions, route details, and truck specifications. You can filter the data by date range, truck ID, or route ID, and get insights into potential delays.
""")

# Sidebar for filter options
st.sidebar.header("Filter Options")
st.sidebar.markdown("Select a filter to customize your data before making predictions:")
filter_type = st.sidebar.radio(
    "Select Filter Type:",
    ['By Date Range', 'By Truck ID', 'By Route ID'],
    help="Choose how you want to filter the data before prediction."
)

# Date Range Filter
if filter_type == 'By Date Range':
    st.sidebar.subheader("Date Range")
    from_date = st.sidebar.date_input(
        "Start Date:", 
        value=min(final_merge['departure_date']),
        help="Select the start date for filtering records."
    )
    to_date = st.sidebar.date_input(
        "End Date:", 
        value=max(final_merge['departure_date']),
        help="Select the end date for filtering records."
    )

# Truck ID Filter
elif filter_type == 'By Truck ID':
    st.sidebar.subheader("Truck ID")
    truck_id = st.sidebar.selectbox(
        'Select Truck ID:', 
        final_merge['truck_id'].unique(),
        help="Choose a specific truck ID to filter records."
    )

# Route ID Filter
elif filter_type == 'By Route ID':
    st.sidebar.subheader("Route ID")
    route_id = st.sidebar.selectbox(
        'Select Route ID:', 
        final_merge['route_id'].unique(),
        help="Choose a specific route ID to filter records."
    )

# Main section: Prediction and results
st.markdown("### Prediction Results")

# Button to trigger prediction
if st.sidebar.button("Predict"):
    try:
        # Apply the appropriate filter based on the user's selection
        if filter_type == 'By Date Range':
            filtered_data = final_merge[(final_merge['departure_date'] >= str(from_date)) & 
                                        (final_merge['departure_date'] <= str(to_date))].copy()
        elif filter_type == 'By Truck ID':
            filtered_data = final_merge[final_merge['truck_id'] == truck_id].copy()
        elif filter_type == 'By Route ID':
            filtered_data = final_merge[final_merge['route_id'] == route_id].copy()

        # Display a preview of the filtered data
        st.write("### Sample of Filtered Data")
        st.dataframe(filtered_data.head(10))

        # Drop columns not needed for prediction
        if 'delay' in filtered_data.columns:
            filtered_data = filtered_data.drop(columns=['delay', 'unique_id'])

        # Preprocess the data: select continuous, categorical, and object columns
        cts_cols = ['route_avg_temp', 'route_avg_wind_speed', 'route_avg_precip', 'route_avg_humidity', 
                    'route_avg_visibility', 'route_avg_pressure', 'distance', 'average_hours', 'temp_origin', 
                    'wind_speed_origin', 'precip_origin', 'humidity_origin', 'visibility_origin', 'pressure_origin',
                    'temp_destination', 'wind_speed_destination', 'precip_destination', 'humidity_destination', 
                    'visibility_destination', 'pressure_destination', 'avg_no_of_vehicles', 'truck_age', 
                    'load_capacity_pounds', 'mileage_mpg', 'age', 'experience', 'average_speed_mph']
        
        encode_columns = ['route_description', 'description_origin', 'description_destination', 'fuel_type', 'gender', 'driving_style']
        object_cols = ['accident', 'ratings', 'is_midnight']
        
        # Convert object columns to numeric
        filtered_data[object_cols] = filtered_data[object_cols].apply(pd.to_numeric, errors='coerce')

        # Prepare feature data (X)
        X_filtered = filtered_data[cts_cols + object_cols + encode_columns].copy()
        X_filtered.reset_index(drop=True, inplace=True)

        # Apply encoding and scaling
        X_filtered_encoded = encoder.transform(X_filtered[encode_columns])
        encoded_filtered_df = pd.DataFrame(X_filtered_encoded, columns=encoder.get_feature_names_out(encode_columns))
        X_filtered_final = pd.concat([X_filtered.drop(columns=encode_columns).reset_index(drop=True), encoded_filtered_df], axis=1)
        X_filtered_final[cts_cols] = scaler.transform(X_filtered_final[cts_cols])

        # Make predictions using the loaded model
        predictions = model.predict(X_filtered_final)
        filtered_data['Predicted Delay'] = predictions

        # Display the results
        st.success(f"Predictions complete! Found {len(filtered_data)} records.")
        st.write("### Prediction Results")
        st.dataframe(filtered_data[['truck_id', 'route_id', 'departure_date', 'Predicted Delay']])

        # Download option for predictions
        st.download_button(
            label="Download Predictions as CSV",
            data=filtered_data.to_csv(index=False),
            file_name='truck_delay_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    
    st.info("🚛 Ready to explore the data? Adjust the filters on the left to zero in on your desired insights, then hit 'Predict' to uncover potential delays!")

