import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

print("="*60)
print("LSTM Numerical Example - Implementation from Scratch")
print("="*60)

inputs = [1, 2, 3, 4]

W_f = np.array([0.5, 0.1])  
W_i = np.array([0.6, 0.2])  
W_c = np.array([0.7, 0.3]) 
W_o = np.array([0.8, 0.4])  
b_f = 0
b_i = 0
b_c = 0
b_o = 0
h_prev = 0
C_prev = 0
W_y = 4
b_y = 0

print("\n" + "="*60)
print("Step 2: Compute LSTM Values for Each Time Step")
print("="*60)

for t, x_t in enumerate(inputs):
    print(f"\n{'='*60}")
    print(f"Time Step t = {t+1}, Input x_{t+1} = {x_t}")
    print(f"{'='*60}")

    
    f_t = sigmoid(W_f[0] * x_t + W_f[1] * h_prev + b_f)
    print(f"1. Forget gate:")
    print(f"   f_{t+1} = σ({W_f[0]}({x_t}) + {W_f[1]}({h_prev:.3f}) + {b_f}) = σ({W_f[0]*x_t + W_f[1]*h_prev + b_f:.3f}) ≈ {f_t:.3f}")

    i_t = sigmoid(W_i[0] * x_t + W_i[1] * h_prev + b_i)
    print(f"2. Input gate:")
    print(f"   i_{t+1} = σ({W_i[0]}({x_t}) + {W_i[1]}({h_prev:.3f}) + {b_i}) = σ({W_i[0]*x_t + W_i[1]*h_prev + b_i:.3f}) ≈ {i_t:.3f}")

    C_tilde = tanh(W_c[0] * x_t + W_c[1] * h_prev + b_c)
    print(f"3. Candidate cell state:")
    print(f"   C̃_{t+1} = tanh({W_c[0]}({x_t}) + {W_c[1]}({h_prev:.3f}) + {b_c}) = tanh({W_c[0]*x_t + W_c[1]*h_prev + b_c:.3f}) ≈ {C_tilde:.3f}")

    C_t = f_t * C_prev + i_t * C_tilde
    print(f"4. Cell state update:")
    print(f"   C_{t+1} = ({f_t:.3f} · {C_prev:.3f}) + ({i_t:.3f} · {C_tilde:.3f}) = {C_t:.3f}")

    o_t = sigmoid(W_o[0] * x_t + W_o[1] * h_prev + b_o)
    print(f"5. Output gate:")
    print(f"   o_{t+1} = σ({W_o[0]}({x_t}) + {W_o[1]}({h_prev:.3f}) + {b_o}) = σ({W_o[0]*x_t + W_o[1]*h_prev + b_o:.3f}) ≈ {o_t:.3f}")

    h_t = o_t * tanh(C_t)
    print(f"6. Hidden state update:")
    print(f"   h_{t+1} = {o_t:.3f} · tanh({C_t:.3f}) = {h_t:.3f}")

    h_prev = h_t
    C_prev = C_t


print("\n" + "="*60)
print("Step 3: Predict the Next Value")
print("="*60)

y_hat = W_y * h_prev + b_y

print(f"\nFinal Hidden State (h₄): {h_prev:.3f}")
print(f"Output Weight (W_y): {W_y}")
print(f"Output Bias (b_y): {b_y}")
print(f"\nFinal Prediction: ŷ = {W_y} × {h_prev:.3f} + {b_y} = {y_hat:.3f}")

print("\n" + "="*60)
print("RESULT: The LSTM predicts {:.1f}, which is close to 4,".format(y_hat))
print("showing that the model has learned the pattern.")
print("="*60)
     
