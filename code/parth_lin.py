import numpy as np
from getLinF16 import getLinF16

## Coordinated Turn
##         Vt   h    gamma    psidot        thetadot
inputs = [502.0, 0.0, 0.0, 0.3*57.29577951, 0.0]
xcg = 0.3
orient = 3      # Coordinated Turn
printOn = False
A, B = getLinF16(orient, inputs, xcg, printOn)

# print(A)

            # states => vt, alpha, beta, p, q, r, theta, phi
A_exp = np.array([
    [-0.09, -169.0, 31.4, -7.73, -31.2, 5e-4, -7.75, 2e-3],                    
    [-5e-4, -1.05, 3e-4, -0.0607, 0.0151, -5e-4, 0.903, -1e-4],            
    [-1e-4, 1.4e-4, -3.22e-7, 1.3e-2, -0.0032, 2.48e-1, 7e-6, -9.61e-1],
    [.0, .0, .0, .0, 0.3,  1.0, 0.0508, 0.0105],       
    [.0, .0, .0, -0.3, .0, .0, 0.203, -0.979],                   
    [-3e-4, 0.0578, -5.94e1, .0, .0, -3.19, -0.0469, 1.64],   
    [1e-3, 1.26, 1e-3, .0, .0, 0.0589, -1.66, -0.0175],     
    [5e-4, -0.617, 8.88, .0, .0, -0.299, 0.0123, -0.565], 

])

        # index             0     1    2      3      4   6  7  8
        # default states => vt, alpha, beta, phi, theta, p, q, r                    # some serious problems # completely redo both matrix 
A1 = np.hstack((A[0:5, 0:5], A[0:5, 6:9]))          
A2 = np.hstack((A[6:9, 0:5], A[6:9, 6:9]))
A_ = np.vstack((A1, A2))
# print(A_)

print(np.abs(A_ - A_exp))

import pandas as pd
A_matrix = pd.DataFrame(A_)
A_matrix.to_csv("A_.csv")

A_matrix = pd.DataFrame(A_exp)
A_matrix.to_csv("A_exp.csv")

# import pandas as pd
Adiff = pd.DataFrame(np.abs(A_ - A_exp))
Adiff.to_csv("Adiff.csv")

x_states = [ 5.02000000e+02,  2.48485146e-01,  0.00000000e+00,  1.36034701e+00,
  5.29542475e-02, -1.58788507e-02,  2.92969899e-01,
 6.25819498e-02]            # vt alpha beta phi theta p q r
x_dot = np.matmul(A_ ,x_states)
print(x_dot)
# [-1.01973914e+02 -3.46938785e-01 -1.08969473e-01  1.58353367e-02
#  -4.07528901e-01  2.78760839e-02  3.16495493e-01 -1.52888171e-02]

x_states_exp = [ 502.0, 0.2485, 4.8e-4, 1.267, 0.05185, -0.01555, 0.2934, 0.06071]            # vt alpha beta phi theta p q r
x_dot_exp = np.matmul(A_exp ,x_states_exp)
print(x_dot_exp)
# [-1.00846794e+02 -3.23106917e-01 -9.60567864e-02  1.55471750e-02
#  -3.79974890e-01 -2.93402600e-02  3.26088160e-01  7.58950200e-02]

x_dot_diff = abs(x_dot - x_dot_exp)
print(x_dot_diff)

