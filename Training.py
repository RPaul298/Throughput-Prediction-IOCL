import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout, Attention, Concatenate, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os
import joblib
import keras_tuner as kt # ## Import KerasTuner

# --- 1. Data Loading and Preparation (No Changes) ---
print("Loading and preparing data...")
try:
    inp_df = pd.read_csv("input_data.csv")
    out_df = pd.read_csv("output_data.csv")
except FileNotFoundError:
    print("Error: Make sure 'input_data.csv' and 'output_data.csv' are in the same directory.")
    exit()

# (Data preparation code is identical to your original script)
pressure = inp_df['Pressure'].values
temperature = inp_df['Temperature'].values
Riser_Steam_rate=inp_df['Riser Steam Rate'].values
catalyst_type=inp_df['Catalyst Type'].values
catalyst_age=inp_df['Catalyst Age'].values
catalyst_Amount = inp_df['Catalyst-to-oil Ratio'].values
crude_density = inp_df['CrudeDensity'].values
sulfur_c=inp_df['Sulfur Content'].values
nitrogen_c=inp_df['Nitrogen Content'].values
ccr_c=inp_df['Conradson Carbon Residue Content'].values
temp_cat = temperature * catalyst_Amount
press_temp = pressure / (temperature + 1e-6)
raw_features = np.column_stack([
    pressure, temperature, Riser_Steam_rate, catalyst_type, catalyst_age,
    catalyst_Amount, crude_density, sulfur_c, nitrogen_c, ccr_c,
    temp_cat, press_temp
])
raw_labels = np.column_stack([
    out_df['Petrol'].values, out_df['Diesel'].values, out_df['Coke'].values,
    out_df['LPG'].values, out_df['Bitumen'].values, out_df['Waste'].values
])
log_labels = np.log1p(raw_labels)

feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()
n_features = feature_scaler.fit_transform(raw_features)
n_labels = label_scaler.fit_transform(log_labels)

t_s = 168
f_s = 168

X_seq, Y_seq = [], []
print("Creating sequences...")
for i in range(len(n_features) - t_s - f_s + 1):
    X_seq.append(n_features[i:i + t_s])
    Y_seq.append(n_labels[i + t_s:i + t_s + f_s])
X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=45)
print(f"Training data shape: {Xtrain.shape}")
print(f"Testing data shape: {Xtest.shape}")

# --- Custom Weighted Loss Function (No Changes) ---
o_w=tf.constant([2.0, 2.0, 4.0, 3.0, 1.0, 5.0])
def custom_weight(y_true,y_pred):
    error=tf.abs(y_true-y_pred)
    w_error=error*o_w
    return tf.reduce_mean(w_error)

# --- 4. Model Builder for Hyperparameter Tuning ---
## We create a function that builds the model. KerasTuner will call this function.
# --- 4. Model Builder with Fixes ---
def build_model(hp):
    # --- Define Hyperparameters to Tune ---
    lstm_units = hp.Int('lstm_units', min_value=80, max_value=200, step=40)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.4, step=0.1)
    # ## I'm keeping the high learning rate here, as gradient clipping should handle it.
    learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001])

    input_shape = (Xtrain.shape[1], Xtrain.shape[2])
    output_shape = ytrain.shape[2]

    # --- Encoder with Bidirectional LSTM ---
    encoder_inputs = Input(shape=input_shape)
    # ## FIX: Removed activation='relu' to use the default 'tanh'
    e_bilstm_1 = Bidirectional(LSTM(lstm_units, return_sequences=True))(encoder_inputs)
    e_dropout_1 = Dropout(dropout_rate)(e_bilstm_1)

    # ## FIX: Removed activation='relu'
    encoder_outputs, fwd_h, fwd_c, bwd_h, bwd_c = Bidirectional(
        LSTM(lstm_units, return_state=True, return_sequences=True)
    )(e_dropout_1)

    # Concatenate states
    state_h = Concatenate()([fwd_h, bwd_h])
    state_c = Concatenate()([fwd_c, bwd_c])
    encoder_states = [state_h, state_c]

    # --- Decoder ---
    decoder_lstm_units = lstm_units * 2
    decoder_inputs = RepeatVector(f_s)(state_h)

    # ## FIX: Removed activation='relu'
    d_lstm_1 = LSTM(decoder_lstm_units, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
    d_dropout_1 = Dropout(dropout_rate)(d_lstm_1)
    # ## FIX: Removed activation='relu'
    decoder_outputs_lstm = LSTM(decoder_lstm_units, return_sequences=True)(d_dropout_1)

    # --- Attention Mechanism ---
    attention_layer = Attention()
    context_vector = attention_layer([decoder_outputs_lstm, encoder_outputs])
    combined_output = Concatenate()([decoder_outputs_lstm, context_vector])
    outputs = TimeDistributed(Dense(output_shape))(combined_output)

    model = Model(encoder_inputs, outputs)

    # ## FIX: Added 'clipvalue=1.0' to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)

    model.compile(optimizer=optimizer, loss=custom_weight)

    return model

# --- 5. Hyperparameter Search ---
## Instantiate the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,  # Number of hyperparameter combinations to try
    executions_per_trial=1, # Number of models to train per combination
    directory='tuning_dir',
    project_name='fcc_throughput_tuning'
)

tuner.search_space_summary()

print("\n--- Starting Hyperparameter Search ---")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Start the search
tuner.search(
    Xtrain, ytrain,
    validation_data=(Xtest, ytest),
    epochs=50, # Use fewer epochs for the search to save time
    batch_size=32,
    callbacks=[early_stop]
)

# --- Retrieve and Summarize the Best Model ---
print("\n--- Hyperparameter Search Complete ---")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
Best Hyperparameters Found:
- LSTM Units: {best_hps.get('lstm_units')}
- Dropout Rate: {best_hps.get('dropout_rate')}
- Learning Rate: {best_hps.get('learning_rate')}
""")

# Get the optimal model
model = tuner.get_best_models(num_models=1)[0]
model.summary()

# Save the best model
MODEL_PATH = "Throughput_Prediction_Optimized_002.h5"
model.save(MODEL_PATH)
print(f"\nBest model saved to {MODEL_PATH}")


# --- 6. Evaluation (Using the best model found) ---
print("\n--- Starting Evaluation of Best Model ---")
y_pred = model.predict(Xtest)

# (Evaluation code is identical to your original script)
ytest_flat = ytest.flatten()
y_pred_flat = y_pred.flatten()
plt.figure(figsize=(8, 8))
plt.scatter(ytest_flat, y_pred_flat, alpha=0.6, s=10, label='Model Predictions')
plt.plot([min(ytest_flat), max(ytest_flat)], [min(ytest_flat), max(ytest_flat)], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Values (Scaled)"), plt.ylabel("Predicted Values (Scaled)"), plt.title("Prediction vs. Actual Values")
plt.grid(True), plt.legend(), plt.show()

residuals = ytest_flat - y_pred_flat
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_flat, residuals, alpha=0.6, s=10)
plt.axhline(y=0, color='r', linestyle='--'), plt.xlabel("Predicted Values (Scaled)"), plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs. Predicted Values"), plt.grid(True), plt.show()

print("\n--- Analyzing Error in Real-World Units ---")
num_samples, seq_len, num_outputs = y_pred.shape
y_pred_reshaped = y_pred.reshape(num_samples * seq_len, num_outputs)
ytest_reshaped = ytest.reshape(num_samples * seq_len, num_outputs)
y_pred_log = label_scaler.inverse_transform(y_pred_reshaped)
ytest_log = label_scaler.inverse_transform(ytest_reshaped)
y_pred_real = np.expm1(y_pred_log)
ytest_real = np.expm1(ytest_log)
real_mae = mean_absolute_error(ytest_real, y_pred_real)
print(f"\nOverall Mean Absolute Error (Real-World Scale): {real_mae:.4f}")

print("\n--- Performance Breakdown by Output ---")
output_labels = ['Petrol', 'Diesel', 'Coke', 'LPG', 'Bitumen', 'Waste']
for i in range(num_outputs):
    per_output_mae = mean_absolute_error(ytest_real[:, i], y_pred_real[:, i])
    print(f"  -> MAE for {output_labels[i]}: {per_output_mae:.4f}")

joblib.dump(feature_scaler,"x_scaler.gz")
joblib.dump(label_scaler,"y_scaler.gz")
print("\nScalers are saved for future use in prediction as x_scaler.gz and y_scaler.gz")