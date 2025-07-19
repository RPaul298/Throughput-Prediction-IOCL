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
model = load_model("Throughput_Prediction_001.h5", compile=False)
s_x = joblib.load("x_scaler.gz")
s_y = joblib.load("y_scaler.gz")

# --- 2. Define Constants ---
ts = 168  # Timesteps in
fs = 168  # Timesteps out

# Define the number of features the user controls vs. what the model needs
num_primary_features = 5  # Pressure, Temp, Catalyst_Type,catalyst amount, Crude Density
num_model_features = 7    # All features the model was trained on

product_names = ["Petrol", "Diesel", "Coke", "LPG", "Bitumen"]
output_labels = ['Petrol', 'Diesel', 'Coke', 'LPG', 'Bitumen', 'Waste']


# --- 3. Define the Objective Function Factory ---
def max_throughput(product_index):
    """
    This function now takes the 4 primary inputs from the optimizer,
    calculates the 2 engineered features, and then runs the model.
    """
    def objective_function(primary_inputs):
        # primary_inputs is a NumPy array with 4 values from the optimizer
        pressure, temp, catalyst_t,catalyst_r, crude_density = primary_inputs
        
        # Manually calculate the engineered features
        temp_cat = temp * catalyst_r
        press_temp = pressure / (temp + 1e-6) # Use epsilon for safety
        
        # Combine into the full 6-feature set that the model expects
        full_feature_set = np.array([
            pressure, temp, catalyst_t,catalyst_r, crude_density, temp_cat, press_temp
        ])
        
        # Prepare the sequence for the model
        input_sequence = np.tile(full_feature_set, (ts, 1)).reshape(1, ts, num_model_features)
        
        # Predict and calculate total yield
        predicted_output = model.predict(input_sequence, verbose=0)[0]
        total_yield = np.sum(predicted_output[:, product_index])
        
        return -total_yield # Return negative yield for minimization

    return objective_function


# --- 4. Main Optimization Orchestrator ---
def func2(product_to_maximize):
    if product_to_maximize not in product_names:
        raise ValueError("Product not valid for optimization.")
    
    idx = output_labels.index(product_to_maximize)
    print(f"\nOptimizing for: {product_to_maximize} (Index {idx})")
    
    # The optimizer only controls the 4 primary features
    bounds = [(0, 1)] * num_primary_features
    initial_guess = [0.5] * num_primary_features
    
    # Get the objective function tailored for the desired product
    objective_fx = max_throughput(idx)
    
    # Run the optimization
    result = minimize(
        objective_fx,
        x0=initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    if result.success:
        print('\n✅ Optimization Succeeded')
        
        # Optimal primary features found by the optimizer (scaled)
        opt_primary_scaled = result.x
        
        # --- Reconstruct the full feature set to inverse transform ---
        p, t, c_type,c_ratio, cd = opt_primary_scaled
        tc = t * c_ratio
        pt = p / (t + 1e-6)
        opt_full_scaled = np.array([p, t, c_type,c_ratio, cd, tc, pt]).reshape(1, -1)
        
        # Inverse transform to get real-world values
        opt_full_real = s_x.inverse_transform(opt_full_scaled)[0]

        # --- Get the final predicted yield in real-world units ---
        input_seq = np.tile(opt_full_scaled, (ts, 1)).reshape(1, ts, num_model_features)
        pred_scaled_output = model.predict(input_seq, verbose=0)[0]
        pred_real_output = s_y.inverse_transform(pred_scaled_output)
        
        # Inverse the log transform to get the final real values
        final_real_output = np.expm1(pred_real_output)
        total_real_yield = np.sum(final_real_output[:, idx])
        
        print("\n--- Optimal Input Conditions ---")
        print(f"Scaled Primary Inputs: {np.round(opt_primary_scaled, 4)}")
        print("\nReal-World Primary Inputs:")
        print(f"  Pressure: {opt_full_real[0]:.2f}")
        print(f"  Temperature: {opt_full_real[1]:.2f}")
        print(f"  Catalyst Type: {opt_full_real[2]:.2f}")            
        print(f"  Catalyst-to-Oil Ratio: {opt_full_real[3]:.2f}")
        print(f"  Crude Density: {opt_full_real[4]:.2f}")

        print("\n--- Predicted Maximum Yield ---")
        print(f"📦 Total {product_to_maximize} Yield: {total_real_yield:.2f} (real-world units) over {fs} hours")
        
        return opt_full_real, total_real_yield
    else:
        print("\n❌ Optimization Failed")
        print(f"Message: {result.message}")
        return None, None

# --- 5. Run the Optimization ---
if __name__ == "__main__":
    product_to_max = "Diesel" 
    func2(product_to_max)