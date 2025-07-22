import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
import tensorflow as tf

# --- 0. Define Custom Loss Function (Required for Loading the Model) ---
# Keras needs the definition of any custom objects used during training.
o_w = tf.constant([2.0, 2.0, 4.0, 3.0, 1.0, 5.0])
def custom_weight(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    w_error = error * o_w
    return tf.reduce_mean(w_error)

# --- 1. Load All Saved Artifacts ---
print("Loading model and scalers...")
try:
    model = load_model(
        "Throughput_Prediction_001.h5",
        custom_objects={'custom_weight': custom_weight},
        compile=False
    )
    feature_scaler = joblib.load("x_scaler.gz")
    label_scaler = joblib.load("y_scaler.gz")
    # Load the original input data to get a sample
    inp_df = pd.read_csv("input_data.csv")
except FileNotFoundError as e:
    print(f"Error: Could not find a required file. Make sure all files are in the same directory.")
    print(e)
    exit()

print("Model and scalers loaded successfully.")

# --- 2. Prepare Input Data for Prediction ---
# We need 168 hours (7 days) of input data to make a prediction.
# For this example, we'll take the first 168 rows from our original dataset.
# In a real application, this would be the MOST RECENT 168 hours of plant data.
t_s = 168 # Timesteps

# Create raw features just like in the training script
raw_input_sample = inp_df.head(t_s)
pressure = raw_input_sample['Pressure'].values
temperature = raw_input_sample['Temperature'].values
Riser_Steam_rate = raw_input_sample['Riser Steam Rate'].values
catalyst_type = raw_input_sample['Catalyst Type'].values
catalyst_age = raw_input_sample['Catalyst Age'].values
catalyst_Amount = raw_input_sample['Catalyst-to-oil Ratio'].values
crude_density = raw_input_sample['CrudeDensity'].values
sulfur_c = raw_input_sample['Sulfur Content'].values
nitrogen_c = raw_input_sample['Nitrogen Content'].values
ccr_c = raw_input_sample['Conradson Carbon Residue Content'].values
temp_cat = temperature * catalyst_Amount
press_temp = pressure / (temperature + 1e-6)

features_to_scale = np.column_stack([
    pressure, temperature, Riser_Steam_rate, catalyst_type, catalyst_age,
    catalyst_Amount, crude_density, sulfur_c, nitrogen_c, ccr_c,
    temp_cat, press_temp
])

# Scale the features using the loaded scaler
scaled_features = feature_scaler.transform(features_to_scale)

# Reshape the data for the model: (1, timesteps, features)
# The '1' is the batch size.
input_for_model = np.expand_dims(scaled_features, axis=0)
print(f"Input data prepared with shape: {input_for_model.shape}")

# --- 3. Make Prediction ---
print("\nMaking prediction for the next 7 days (168 hours)...")
scaled_prediction = model.predict(input_for_model)

# The prediction shape will be (1, 168, 6)

# --- 4. Process the Output ---
# Reshape for inverse transform: from (1, 168, 6) to (168, 6)
scaled_prediction_reshaped = scaled_prediction.reshape(t_s, -1)

# Inverse the scaling to get back to the log-transformed scale
log_prediction = label_scaler.inverse_transform(scaled_prediction_reshaped)

# Inverse the log transform to get the final real-world values
final_prediction = np.expm1(log_prediction)

# --- 5. Display the Results ---
output_labels = ['Petrol', 'Diesel', 'Coke', 'LPG', 'Bitumen', 'Waste']
predictions_df = pd.DataFrame(final_prediction, columns=output_labels)
predictions_df.index = [f"Hour_{i+1}" for i in range(len(predictions_df))]

print("\n--- Predicted Yields ---")
print(predictions_df.head(10)) # Display predictions for the first 10 hours