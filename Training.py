import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os
import joblib

# --- 1. Data Loading and Preparation ---
print("Loading and preparing data...")
try:
    inp_df = pd.read_csv("input_data.csv")
    out_df = pd.read_csv("output_data.csv")
except FileNotFoundError:
    print("Error: Make sure 'input_data.csv' and 'output_data.csv' are in the same directory.")
    exit()

pressure = inp_df['Pressure'].values
temperature = inp_df['Temperature'].values
catalyst = inp_df['Catalyst'].values
crude_density = inp_df['CrudeDensity'].values
# Feature Engineering: Create interactive features
temp_cat = temperature * catalyst
press_temp = pressure / (temperature + 1e-6) # Add epsilon to avoid division by zero

# Combine all features
raw_features = np.column_stack([
    pressure,
    temperature,
    catalyst,
    crude_density,
    temp_cat,      # Interactive feature
    press_temp     # Interactive feature
])

# Combine all output labels
raw_labels = np.column_stack([
    out_df['Petrol'].values,
    out_df['Diesel'].values,
    out_df['Coke'].values,
    out_df['LPG'].values,
    out_df['Bitumen'].values,
    out_df['Waste'].values
])

# Apply log transform to labels to normalize distribution and handle potential skewness
log_labels = np.log1p(raw_labels)

# --- 2. Scaling and Sequencing ---
# Using separate scalers for features and labels is good practice.
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()


# Fit and transform the data
n_features = feature_scaler.fit_transform(raw_features)
n_labels = label_scaler.fit_transform(log_labels)

t_s = 168  # Timesteps in input sequence (history)
f_s = 168  # Timesteps to forecast (future)

X_seq = []
Y_seq = []

print("Creating sequences...")
# We use t_s steps of history to predict f_s steps in the future
for i in range(len(n_features) - t_s - f_s + 1):
    X_seq.append(n_features[i:i + t_s])
    Y_seq.append(n_labels[i + t_s:i + t_s + f_s])

X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)

print(f"Generated {len(X_seq)} sequences.")

# --- 3. Train/Test Split ---
Xtrain, Xtest, ytrain, ytest = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=45)
print(f"Training data shape: {Xtrain.shape}")
print(f"Testing data shape: {Xtest.shape}")
print(f"Training labels shape: {ytrain.shape}")
print(f"Testing labels shape: {ytest.shape}")

# --- 4. Model Loading or Creation ---
print("\nLoading or creating model...")
MODEL_PATH = "Throughput_Prediction_001.h5"
model = None

if os.path.exists(MODEL_PATH):
    try:
        # Load the model and check if the input shape matches the new data
        print("Pre-trained model file found. Checking for compatibility...")
        temp_model = load_model(MODEL_PATH, compile=False)
        # Check if the model's expected input features match our current data
        if temp_model.input_shape[2] == Xtrain.shape[2]:
            print("Model is compatible. Loading pre-trained model.")
            model = temp_model
        else:
            print(f"Model incompatibility detected. Expected {temp_model.input_shape[2]} features but data has {Xtrain.shape[2]}.")
            print("A new model will be created.")
    except Exception as e:
        print(f"Could not load or validate pre-trained model due to an error: {e}")
        print("A new model will be created.")

if model is None:
    print("Creating a new Encoder-Decoder model.")
    # Define a robust Encoder-Decoder model for sequence-to-sequence forecasting
    # Input shape: (timesteps, features)
    input_shape = (Xtrain.shape[1], Xtrain.shape[2])
    output_shape = ytrain.shape[2]

    # Encoder
    encoder_inputs = Input(shape=input_shape)
    # First Layer
    e_lstm_1 = LSTM(150, activation='relu', return_sequences=True)(encoder_inputs)
    e_dropout_1 = Dropout(0.2)(e_lstm_1)  # 20% of the neurons are turned off to prevent overfitting
    # Second and Final Layer, returning state
    _, state_h, state_c = LSTM(150, activation='relu', return_state=True)(e_dropout_1)
    encoder_states = [state_h, state_c]

    # The RepeatVector layer takes the encoder's context and
    # repeats it f_s times to provide it as input to each step of the decoder.
    decoder_inputs = RepeatVector(f_s)(state_h) # Use the hidden state for the context

    # Decoder
    # First decoder LSTM layer initialized with encoder context
    d_lstm_1 = LSTM(150, activation='relu', return_sequences=True)(decoder_inputs, initial_state=encoder_states)
    d_dropout_1 = Dropout(0.2)(d_lstm_1) # 20% neurons are turned off
    # Second decoder LSTM layer
    d_lstm_2 = LSTM(150, activation='relu', return_sequences=True)(d_dropout_1)

    # The TimeDistributed layer applies a standard Dense layer to each temporal
    # slice of the input, allowing us to get an output for each of the f_s timesteps.
    outputs = TimeDistributed(Dense(output_shape))(d_lstm_2)

    model = Model(encoder_inputs, outputs)

# --- 5. Model Compilation and Training ---
model.compile(optimizer='adam', loss='mae')
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    Xtrain, ytrain,
    validation_data=(Xtest, ytest),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

model.save(MODEL_PATH)
print("\nModel training complete and saved successfully!")

# ===================================================================
#                      COMPREHENSIVE MODEL EVALUATION
# ===================================================================

print("\n--- Starting Comprehensive Model Evaluation ---")

# --- 6. Get Model Predictions on Test Data ---
y_pred = model.predict(Xtest)

# --- 7. Visual Analysis: Prediction vs. Actual Plot ---
ytest_flat = ytest.flatten()
y_pred_flat = y_pred.flatten()

plt.figure(figsize=(8, 8))
plt.scatter(ytest_flat, y_pred_flat, alpha=0.6, s=10, label='Model Predictions')
plt.plot([min(ytest_flat), max(ytest_flat)], [min(ytest_flat), max(ytest_flat)], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Values (Scaled)")
plt.ylabel("Predicted Values (Scaled)")
plt.title("Prediction vs. Actual Values")
plt.grid(True)
plt.legend()
plt.show()

# --- 8. Visual Analysis: Residuals Plot ---
residuals = ytest_flat - y_pred_flat
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_flat, residuals, alpha=0.6, s=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values (Scaled)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs. Predicted Values")
plt.grid(True)
plt.show()

# --- 9. Analysis in Real-World Units ---
print("\n--- Analyzing Error in Real-World Units ---")
num_samples, seq_len, num_outputs = y_pred.shape
y_pred_reshaped = y_pred.reshape(num_samples * seq_len, num_outputs)
ytest_reshaped = ytest.reshape(num_samples * seq_len, num_outputs)

# Inverse transform to get back to the log scale
y_pred_log = label_scaler.inverse_transform(y_pred_reshaped)
ytest_log = label_scaler.inverse_transform(ytest_reshaped)

# Inverse the log transform (expm1) to get the original real-world scale
y_pred_real = np.expm1(y_pred_log)
ytest_real = np.expm1(ytest_log)

real_mae = mean_absolute_error(ytest_real, y_pred_real)
print(f"\nOverall Mean Absolute Error (Real-World Scale): {real_mae:.4f}")

# --- 10. Per-Output Performance Analysis ---
print("\n--- Performance Breakdown by Output ---")
output_labels = ['Petrol', 'Diesel', 'Coke', 'LPG', 'Bitumen', 'Waste']
for i in range(num_outputs):
    per_output_mae = mean_absolute_error(ytest_real[:, i], y_pred_real[:, i])
    print(f"  -> MAE for {output_labels[i]}: {per_output_mae:.4f}")
#Save fitted scaler for better use in predictions and optimization
joblib.dump(feature_scaler,"x_scaler.gz")
joblib.dump(label_scaler,"y_scaler.gz")
print("Scalers are saved for future use in prediction as x_scaler.gz and y_scaler.gz")