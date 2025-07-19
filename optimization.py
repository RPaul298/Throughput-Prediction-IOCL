import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import joblib 

np.random.seed(45)
tf.random.set_seed(45)

model=tf.keras.models.load_model("Throughput_Prediction_001.h5",compile=False)

ts= 168       # How many past hours to use for prediction
fs= 168       # How many future hours to predict 
num_primary_features=4  #Pressure,Temp,Catalyst,Crude Density
num_model_features = 6  # Input features : Pressure, Temp, Catalyst,  Crude Density,catalyst*temp,pressure/(temp + 1e-6)
num_o= 6     # Outputs : Petrol , Diesel, Coke, LPG, Bitumen, Waste 

product_names=["Petrol","Diesel","Coke","LPG","Bitumen"]
output_labels=["Petrol","Diesel","Coke","LPG","Bitumen","Waste"]

s_x=joblib.load("x_scaler.gz")
s_y=joblib.load("y_scaler.gz")

#s_x.fit()
#s_y.fit()

def max_throughput(x):
    def func1(x1):
        i_s=np.tile(x1, (ts,1)).reshape(1, ts, num_features)  #i_s stands for input sequence
        p_o=model.predict(i_s,verbose=0)[0] # p_o stands for product output
        p_y=np.sum(p_o[:,x])
        return -p_y
    
    return func1

def func2(x2):
    if x2 not in product_names:
        raise ValueError(...)
    idx=product_names.index(x2)
    print(f"\n Optimizing for: {x2}(Index{idx})")
    
    bounds=[(0,1)]*num_features
    i_g=[0.5]*num_features
    
    objective_fx=max_throughput(idx)
    
    result=minimize(
        objective_fx,
        x0=i_g,
        bounds=bounds,
        method='L-BFGS-B'
                    )
    if result.success:
        opt_i_scaled=result.x.reshape(1,-1)
        opt_i_real=s_x.inverse_transform(opt_i_scaled)
        input_seq=np.tile(opt_i_scaled,(ts,1)).reshape(1,ts,num_features)
        pred_scaled_output=model.predict(input_seq, verbose=0)[0]
        pred_real_output=s_y.inverse_transform(pred_scaled_output)
        total_real_yield=np.sum(pred_real_output[:, idx])
        total_n_yield=-result.fun 
        
        print('Optimization Succeeded ')
        print('Normalized optimal input (scaled):',opt_i_scaled[0]) 
        print("Real world input",opt_i_real[0])
        print(f"📦 Predicted Total {x2} Yield: {total_n_yield:.2f} units over {fs} hours")
        print(f"Total {x2} yield:{total_real_yield:.2f} over {fs} hours")  
        return opt_i_real, total_real_yield
    else : 
        print("Optimized Failed")
        return None, None 
if __name__=="__main__":
    product_to_max="Diesel" 
    func2(product_to_max)
