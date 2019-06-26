'''
Stanley Bak
F16 GCAS in Python
Compute lineaized version (A, B, C, D matrices) for F-16
'''

from subf16_model import subf16_model
from util import print_matrix
import numpy as np

def jacobFun(Xequil, Uequil, Xcg, printOn, model, adjust_cy, C_and_D):
    '''
    numerically calculates the linearized A, B, C, & D matrices
    of an F-16 model given certain parameters where:

    x = A x  +  B u
    y = C x  +  D u

    There are 13 state variables, 4 inputs, and 10 outputs, making the matrix sizes:
    A: 13x13, B: 13x4, C: 10x13, D: 10x4

    y (outputs): [ Az q alpha theta Vt Ay p r beta phi]^T
    '''

    if printOn:
        print('Running jacobFun.py')

    xe = Xequil.copy()
    ue = Uequil.copy()
    x = xe.copy()
    u = ue.copy()

    n = len(Xequil)
    m = len(Uequil) 

    tol = 1e-6

    xde, _, _, aze, aye = subf16_model(x, u, Xcg=Xcg, model=model, adjust_cy=adjust_cy)

    #####   A matrix    #####
    dx = 0.01*x
    for i in range(0,n):
        if dx[i] == 0.0:
            dx[i] = 0.1

    last = np.zeros((n,1), dtype=float)
    A = np.zeros((n,n), dtype=float)

    for j in range(0,n):
        xt = x
        for i in range(0,10):
            xt[j] = x[j] + dx[j]
            xd1 = subf16_model(xt, u, Xcg=Xcg, model=model, adjust_cy=adjust_cy)[0]
            xt[j] = x[j] - dx[j]
            xd2 = subf16_model(xt, u, Xcg=Xcg, model=model, adjust_cy=adjust_cy)[0]
            A[:, j] = (np.transpose(xd1.ravel() - xd2.ravel()) / (2*dx[j]))
            if np.max(np.abs(A[:,j] - last) / abs(A[:,j] + 1e-12)) < tol:
                break
            dx[j] = 0.5*dx[j]
            last = A[:,j]
        ## column = j
        iteration = i
        if iteration == 10:
            print(f"not converged on A, column {j}")

    AA = 2*A

    #####   B matrix    #####
    du = 0.01*u

    for i in range(0,m):
        if du[i] == 0.0:
            du[i] = 0.1

    last = np.zeros((n,1), dtype=float)
    B = np.zeros((n,m), dtype=float)

    for j in range(0,m):
        ut = u
        for i in range(0,10):
            ut[j] = u[j] + du[j]
            ut = ut
            xd1 = xd2 = subf16_model(x, ut, Xcg=Xcg, model=model, adjust_cy=adjust_cy)[0]
            ut[j] = u[j] - du[j]
            ut = ut
            xd2 = subf16_model(x, ut, Xcg=Xcg, model=model, adjust_cy=adjust_cy)[0]
            B[:, j] = (np.transpose(xd1.ravel() - xd2.ravel()) / (2*du[j]))
            if np.max(np.abs(B[:,j] - last) / abs(B[:,j] + 1e-12)) < tol:
                break
            dx[j] = 0.5*dx[j]
            du[j] = 0.5*du[j]
            last = B[:,j]
        ## column = j
        iteration = i
        if iteration == 10:
            print(f"not converged on B, column {j}")    
    
    BB = 2*B

    if C_and_D == True:
        #####   C matrix    #####
        CC = np.zeros((10, 13))

        u = ue.copy()
        for i in range(13): # az
            x = xe.copy()
            delta = 0.01
            slope1 = 0
            diff = 1

            if xe[i] == 0:
                delta = 0.5
            else:
                delta = delta * xe[i]

            while diff > tol:
                x[i] = xe[i] + delta
                _, _, _, az, _ = subf16_model(x, u, Xcg=Xcg, model=model, adjust_cy=adjust_cy)

                slope2 = (az - aze) / delta
                diff = abs(slope2 - slope1)
                delta = delta * .1
                slope1 = slope2

                assert diff <= 1e6, 'No convergence when numerically computing az part of C matrix'

            CC[0, i] = slope1 / -32.2

        for i in range(13): # ay
            x = xe.copy()
            delta = 0.01
            slope1 = 0
            diff = 1

            if xe[i] == 0:
                delta = 0.5
            else:
                delta = delta * xe[i]

            while diff > tol:
                x[i] = xe[i] + delta
                _, _, _, _, ay = subf16_model(x, u, Xcg=Xcg, model=model, adjust_cy=adjust_cy)

                slope2 = (ay - aye) / delta
                diff = abs(slope2 - slope1)
                delta = delta * .1
                slope1 = slope2

                assert diff <= 1e6, 'No convergence when numerically computing ay part of C matrix'

            CC[5, i] = slope1 / 32.2

        CC[1, :] = [0, 0, 0, 0, 0, 0, 0, 57.3, 0, 0, 0, 0, 0]
        CC[2, :] = [0, 57.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CC[3, :] = [0, 0, 0, 0, 57.3, 0, 0, 0, 0, 0, 0, 0, 0]
        CC[4, :] = [1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CC[6, :] = [0, 0, 0, 0, 0, 0, 57.3, 0, 0, 0, 0, 0, 0]
        CC[7, :] = [0, 0, 0, 0, 0, 0, 0, 0, 57.3, 0, 0, 0, 0]
        CC[8, :] = [0, 0, 57.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CC[9, :] = [0, 0, 0, 57.3, 0, 0, 0, 0, 0, 0, 0, 0, 0]


        #####   D matrix    #####
        D = np.zeros((10, 4))

        x = xe.copy()
        for i in range(4): #  az
            u = ue.copy()
            delta = 0.01
            slope1 = 0
            diff = 1

            if ue[i] == 0:
                delta = 0.5
            else:
                delta = delta * ue[i]

            while diff > tol:
                u[i] = ue[i] + delta
                _, _, _, az, _ = subf16_model(x, u, Xcg=Xcg, model=model, adjust_cy=adjust_cy)

                slope2 = (az - aze) / delta
                diff = abs(slope2 - slope1)
                delta = delta * .1
                slope1 = slope2

                assert diff <= 1e6, 'No convergence when numerically computing az part of D matrix'

            D[0, i] = slope1 / -32.2

        for i in range(4): # ay
            u = ue.copy()
            delta = 0.01
            slope1 = 0
            diff = 1

            if ue[i] == 0:
                delta = 0.5
            else:
                delta = delta * ue[i]

            while diff > tol:
                u[i] = ue[i] + delta
                _, _, _, _, ay = subf16_model(x, u, Xcg=Xcg, model=model, adjust_cy=adjust_cy)

                slope2 = (ay - aye) / delta
                diff = abs(slope2 - slope1)
                delta = delta * .1
                slope1 = slope2

                assert diff <= 1e6, 'No convergence when numerically computing ay part of D matrix'
            D[5, i] = slope1 / 32.2

        D[1, :] = [0, 0, 0, 0]
        D[2, :] = [0, 0, 0, 0]
        D[3, :] = [0, 0, 0, 0]
        D[4, :] = [0, 0, 0, 0]
        D[6, :] = [0, 0, 0, 0]
        D[7, :] = [0, 0, 0, 0]
        D[8, :] = [0, 0, 0, 0]
        D[9, :] = [0, 0, 0, 0]

        # This was done in getLinF16 originally
        # y = [ Az q alpha theta Vt Ay p r beta phi ]T
        #CC([1:4 6:10], :) = deg2rad(CC([2:4 7:10],:));
        #D([1:4 6:10], :) = deg2rad(D([2:4 7:10],:));

        for matrix in [CC, D]:
            for i in range(10):
                if i in [0, 4, 5]:
                    continue

                matrix[i, :] = np.deg2rad(matrix[i, :])

        return AA, BB, CC, D
    
    return AA, BB

