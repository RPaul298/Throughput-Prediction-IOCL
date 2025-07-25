import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import joblib

np.random.seed(45)
tf.random.set_seed(45)

# --- 1. Load Model and Scalers ---
model = load_model("Throughput_Prediction_Optimized_002.h5", compile=False)
s_x = joblib.load("x_scaler.gz")
s_y = joblib.load("y_scaler.gz")

# --- 2. Define Feature Lists ---
# The full list of primary features in the order the model was trained
ALL_PRIMARY_FEATURES = [
    'Pressure', 'Temperature', 'Riser Steam Rate', 'Catalyst Type', 'Catalyst Age',
    'Catalyst-to-oil Ratio', 'CrudeDensity', 'Sulfur Content', 'Nitrogen Content',
    'Conradson Carbon Residue Content'
]

# Define which features an operator can control vs. which are fixed properties
CONTROLLABLE_FEATURES = ['Pressure', 'Temperature', 'Riser Steam Rate', 'Catalyst-to-oil Ratio']
UNCONTROLLABLE_FEATURES = [
    'Catalyst Type', 'Catalyst Age', 'CrudeDensity', 'Sulfur Content',
    'Nitrogen Content', 'Conradson Carbon Residue Content'
]

ts = 168
fs = 168
num_model_features = 12 # 10 primary + 2 engineered
product_names = ["Petrol", "Diesel", "Coke", "LPG", "Bitumen"]
output_labels = ['Petrol', 'Diesel', 'Coke', 'LPG', 'Bitumen', 'Waste']

# --- 3. Define the Objective Function Factory ---
def max_throughput(product_index, uncontrollable_scaled_values):
    """
    The objective function now takes the controllable variables from the optimizer
    and combines them with the fixed, uncontrollable variables.
    """
    def objective_function(controllable_scaled_inputs):
        # Create a dictionary of all primary features
        all_inputs = {}
        for i, feature_name in enumerate(CONTROLLABLE_FEATURES):
            all_inputs[feature_name] = controllable_scaled_inputs[i]
        for feature_name, value in uncontrollable_scaled_values.items():
            all_inputs[feature_name] = value

        # Assemble the full feature vector in the correct order
        ordered_inputs = [all_inputs[name] for name in ALL_PRIMARY_FEATURES]
        p, temp, steam, c_type, c_age, c_ratio, dens, sulf, nit, ccr = ordered_inputs

        # Manually calculate the 2 engineered features
        temp_x_ratio = temp * c_ratio
        press_div_temp = p / (temp + 1e-6)

        full_feature_set = np.array(ordered_inputs + [temp_x_ratio, press_div_temp])

        # Prepare the sequence and predict
        input_sequence = np.tile(full_feature_set, (ts, 1)).reshape(1, ts, num_model_features)
        predicted_output = model.predict(input_sequence, verbose=0)[0]
        total_yield = np.sum(predicted_output[:, product_index])

        return -total_yield

    return objective_function

# --- 4. Main Optimization Orchestrator ---
def func2(product_to_maximize, uncontrollable_inputs):
    """
    Main function now takes a dictionary of the real-world values for
    uncontrollable features.
    """
    idx = output_labels.index(product_to_maximize)
    print(f"\nOptimizing for: {product_to_maximize} (Index {idx})")
    print("\nUsing fixed uncontrollable inputs:")
    for key, val in uncontrollable_inputs.items():
        print(f"  - {key}: {val}")

    # --- Scale the uncontrollable inputs ---
    # Create a temporary array with all features to scale them correctly
    temp_full_array = np.zeros(num_model_features)
    uncontrollable_scaled = {}
    for feature_name, real_value in uncontrollable_inputs.items():
        feature_index = ALL_PRIMARY_FEATURES.index(feature_name)
        temp_full_array[feature_index] = real_value

    # We need to reshape for the scaler, then extract the scaled values
    scaled_array = s_x.transform(np.tile(temp_full_array, (2,1)))[0] # HACK: tile to avoid scaler warning
    for feature_name in UNCONTROLLABLE_FEATURES:
        feature_index = ALL_PRIMARY_FEATURES.index(feature_name)
        uncontrollable_scaled[feature_name] = scaled_array[feature_index]


    # --- Set up and run optimization for controllable features ---
    num_controllable = len(CONTROLLABLE_FEATURES)
    bounds = [(0, 1)] * num_controllable
    initial_guess = [0.5] * num_controllable

    objective_fx = max_throughput(idx, uncontrollable_scaled)

    result = minimize(objective_fx, x0=initial_guess, bounds=bounds, method='L-BFGS-B')

    if result.success:
        print('\n✅ Optimization Succeeded')
        opt_controllable_scaled = result.x

        # --- Reconstruct the full feature set for reporting ---
        final_all_inputs_scaled = {}
        for i, feature_name in enumerate(CONTROLLABLE_FEATURES):
            final_all_inputs_scaled[feature_name] = opt_controllable_scaled[i]
        final_all_inputs_scaled.update(uncontrollable_scaled)

        ordered_final_scaled = [final_all_inputs_scaled[name] for name in ALL_PRIMARY_FEATURES]
        p, t, steam, c_type, c_age, c_ratio, dens, sulf, nit, ccr = ordered_final_scaled
        t_x_r = t * c_ratio
        p_div_t = p / (t + 1e-6)
        opt_full_scaled = np.array(ordered_final_scaled + [t_x_r, p_div_t]).reshape(1, -1)

        # Inverse transform to get real-world values
        opt_full_real = s_x.inverse_transform(opt_full_scaled)[0]

        # Get final yield
        input_seq = np.tile(opt_full_scaled, (ts, 1)).reshape(1, ts, num_model_features)
        pred_scaled_output = model.predict(input_seq, verbose=0)[0]
        pred_log_output = s_y.inverse_transform(pred_scaled_output)
        final_real_output = np.expm1(pred_log_output)
        total_real_yield = np.sum(final_real_output[:, idx])

        # --- Print Results ---
        print("\n--- Optimal Controllable Settings ---")
        for feature_name in CONTROLLABLE_FEATURES:
            feature_index = ALL_PRIMARY_FEATURES.index(feature_name)
            print(f"  {feature_name}: {opt_full_real[feature_index]:.2f}")

        print("\n--- Predicted Maximum Yield ---")
        print(f"📦 Total {product_to_maximize} Yield: {total_real_yield:.2f} metric tons over {fs} hours")

        return opt_full_real, total_real_yield
    else:
        print("\n❌ Optimization Failed")
        print(f"Message: {result.message}")
        return None, None

# --- 5. HOW TO RUN ---
if __name__ == "__main__":
    # Define the properties of the current feedstock you cannot change
    # These are the REAL-WORLD values
    fixed_inputs = {
        'Catalyst Type': 2.0,
        'Catalyst Age': 50.0, # days
        'CrudeDensity': 28.0, # API
        'Sulfur Content': 0.8, # %wt
        'Nitrogen Content': 0.2,
        'Conradson Carbon Residue Content': 4.5
    }

    product_to_max = "Petrol"
    func2(product_to_max, uncontrollable_inputs=fixed_inputs)