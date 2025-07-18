import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

np.random.seed(45)
tf.random.set_seed(45)

model=tf.keras.models.load_model("Throughput_Prediction_001.h5",compile=False)

ts= 168       # How many past hours to use for prediction
fs= 168       # How many future hours to predict 
num_features= 4  # Input features : Pressure, Temp, Catalyst,  Crude Density
num_o= 6     # Outputs : Petrol , Diesel, Coke, LPG, Bitumen, Waste 

product_names=["Petrol","Diesel","Coke","LPG","Bitumen"]

s_x=MinMaxScaler()
s_y=MinMaxScaler()

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
        opt_i=result.x.reshape(1,-1)
        input_seq=np.tile(opt_i,(ts,1)).reshape(1,ts,num_features)
        pred=model.predict(input_seq, verbose=0)[0]
        opt_y=np.sum(pred[:, idx])
        
        print('Optimization Succeeded ')
        print('Normalized optimal input (scaled):',opt_i) 
        print(f"📦 Predicted Total {x2} Yield: {opt_y:.2f} units over {fs} hours")  
        return opt_i, opt_y
    else : 
        print("Optimized Failed")
        return None, None 
if __name__=="__main__":
    product_to_max="Diesel" 
    func2(product_to_max)
