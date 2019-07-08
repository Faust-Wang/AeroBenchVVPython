'''
Stanley Bak
F16 GCAS in Python
Get linearized version of f16 model about a trim point
'''

import numpy as np
from jacobFun import jacobFun
from trimmerFun import trimmerFun
from util import print_matrix


# get the 4-tuple A, B, C, D for the linearized version of the F-16 about a setpoint'
# Given equilibrium trim and controls, returns a linearized state space 
# model of the F - 16.

def getLinF16(orient, inputs, Xcg, printOn=False, model='stevens', adjust_cy=True, C_and_D=False):
    """
    #  'desc'

    #       lin_f16 = getLinF16( xequil, uequil, printOn )
    #
    #   Inputs:
    #       xequil  -   Equilibrium states (13x1)
    #       uequil  -   Equilibrium control (4x1)
    #       printOn -   If true, prints intermediate data
    #
    #   Outputs:
    #       lin_f16 -   labeled state space model of f16 
    #                   (13 state, 4 control, 10 output) 
    #
    #   x_f16 states:
    #       x_f16(1)  = air speed, VT                           (ft / s)
    #       x_f16(2)  = angle of attack, alpha                  (rad)
    #       x_f16(3)  = angle of sideslip, beta                 (rad)
    #       x_f16(4)  = roll angle, phi                         (rad)
    #       x_f16(5)  = pitch angle, theta                      (rad)
    #       x_f16(6)  = yaw angle, psi                          (rad)
    #       x_f16(7)  = roll rate, P                            (rad / s)
    #       x_f16(8)  = pitch rate, Q                           (rad / s)
    #       x_f16(9)  = yaw rate, R                             (rad / s)
    #       x_f16(10) = northward horizontal displacement, pn   (ft)
    #       x_f16(11) = eastward horizontal displacement, pe    (ft)
    #       x_f16(12) = altitude, h                             (ft)
    #       x_f16(13) = engine thrust dynamics lag state, pow   (lbs)
    #
    #   x_f16 controls:
    #       u(1) = throttle                                     (0 to 1)
    #       u(2) = elevator                                     (rad?) 
    #       u(3) = aileron                                      (rad?)
    #       u(4) = rudder                                       (rad?)
    #
    # <a href = "https: / /github.com / pheidlauf / AeroBenchVV">AeroBenchVV< / a>
    # Copyright: GNU General Public License 2017
    #
    # See also: TRIMMERFUN, JACOBFUN
    """
    
    printOn = printOn
    Xequil, Uequil = trimmerFun(orient, inputs, printOn, Xcg, model, adjust_cy)

    if C_and_D == True:
        A, B, C, D = jacobFun(Xequil, Uequil, Xcg, printOn, model, adjust_cy, C_and_D)
        
        # # y = [ Az q alpha theta Vt Ay p r beta phi ]T
        # C([2:4 7:10], :) = deg2rad(C([2:4 7:10], :))
        # D([2:4 7:10], :) = deg2rad(D([2:4 7:10], :))

    else:
        A, B = jacobFun(Xequil, Uequil, Xcg, printOn, model, adjust_cy, C_and_D)        

    if printOn:
        print("A_matrix = ")
        print_matrix(A)
        print(" ")
        print("B_matrix = ")
        print_matrix(B)
        print(" ")
        if C_and_D == True:
            print("C_matrix = ")
            print_matrix(C)
            print(" ")
            print("D_matrix = ")
            print_matrix(D)
            print(" ")
            
    if C_and_D == True:
        return A, B, C, D
    
    return A, B


def get_lon_A(A_matrix):
    
    A = A_matrix

    A_lon = np.array(
        [
        [A[0,0], A[0,1], A[0,4], A[0,7]],
        [A[1,0], A[1,1], A[1,4], A[1,7]],
        [A[4,0], A[4,1], A[4,4], A[4,7]],
        [A[7,0], A[7,1], A[7,4], A[7,7]]
        ]
    )

    return A_lon

def get_lon_B(B_matrix):

    B = B_matrix

    B_lon = np.array(
        [
        [B[0,0], B[0,1]],
        [B[1,0], B[1,1]],
        [B[4,0], B[4,1]],
        [B[7,0], B[7,1]]
        ]
    )

    return B_lon

def get_lat_A(A_matrix):

    A = A_matrix

    A_lat = np.array(
        [
        [A[2,2], A[2,3], A[2,6], A[2,8]],
        [A[3,2], A[3,3], A[3,6], A[3,8]],
        [A[6,2], A[6,3], A[6,6], A[6,8]],
        [A[8,2], A[8,3], A[8,6], A[8,8]]
        ]
    )

    return A_lat

def get_lat_B(B_matrix):

    B = B_matrix

    B_lat = np.array(
        [
        [B[2,2], B[2,3]],
        [B[3,2], B[3,3]],
        [B[6,2], B[6,3]],
        [B[8,2], B[8,3]]
        ]
    )

    return B_lat

# states =    Vt  alpha  beta  phi theta psi   p     q    r    xe    ye   ze   power
#              0   1       2    3    4    5    6     7    8    9     10    11   12