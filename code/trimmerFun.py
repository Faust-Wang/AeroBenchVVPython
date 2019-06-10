'''
Stanley Bak
TrimmerFun in python

This program numerically calculates the equilibrium state and control vectors of an F-16 model given
certain parameters.

states:                                              controls:
	x1 = Vt		x4 = phi	x7 = p	  x10 = pn			u1 = throttle
	x2 = alpha	x5 = theta	x8 = q	  x11 = pe			u2 = elevator
	x3 = beta	x6 = psi    x9 = r	  x12 = alt		    u3 = aileron
                                      x13 = pow         u4 = rudder
'''

from math import sin, cos
import numpy as np

from scipy.optimize import minimize, fmin

from clf16 import clf16
from conf16 import turn_coord_cons, rate_of_climb_cons
from adc import adc

def trimmerFun(orient, inputs, printOn, Xcg=0.35, model='stevens', adjust_cy=False):
# def trimmerFun(Xguess, Uguess, orient, inputs, printOn, Xcg=0.35, model='stevens', adjust_cy=True):
    'calculate equilibrium state'

    # assert isinstance(Xguess, np.ndarray)
    # assert isinstance(Uguess, np.ndarray)
    # assert isinstance(inputs, np.ndarray)

    Xguess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Uguess = np.array([0.0, 0.0, 0.0, 0.0])

    x = Xguess.copy()
    u = Uguess.copy()

    xcg = Xcg

    if printOn:
        print('------------------------------------------------------------')
        print('Running trimmerFun.py')

    # gamma singam rr  pr   tr  phi cphi sphi thetadot coord stab  orient
    const = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1]
    rtod = 57.29577951

    # orient: 'Wings Level (gamma = 0)','Wings Level (gamma <> 0)','Steady Turn','Steady Pull Up'
    const[11] = orient

    # inputs: [Vt, h, gamm, psidot, thetadot]
    x[0] = inputs[0]
    x[11] = inputs[1]

    if orient == 2:
        gamm = inputs[2]
        const[0] = gamm/rtod
        const[1] = sin(const[0])

    elif orient == 3:
        psidot = inputs[3]
        const[4] = psidot/rtod  # tr = turn 
        tr = const[4]

        gamm = inputs[2]
        const[0] = gamm

        phi = turn_coord_cons(tr, x[1], x[2], x[0], gamma=gamm)
        const[5] = phi
        const[6] = sin(phi)
        const[7] = cos(phi)
        x[4] = rate_of_climb_cons(gamm, x[1], x[2], phi)

    elif orient == 4:
        thetadot = inputs[4]
        const[8] = thetadot/rtod

    if orient == 3:
        s = np.zeros(shape=(5,))
        s[0] = x[1]
        s[1] = u[0]
        s[2] = u[1]
        s[3] = u[2]
        s[4] = u[3]
    else:               # for orient 1, 2, 4
        s = np.zeros(shape=(3,))
        s[0] = u[0]
        s[1] = u[1]
        s[2] = x[1]

    if printOn:
        print(f"initial cost = {clf16(s, x, u, xcg, const, model, adjust_cy)}")

    # #=== MINIMIZE Algorithm =============================================================
    # maxiter = 1000
    # tol = 1e-7
    # minimize_tol = 1e-9 #1e-9

    # res = minimize(clf16, s, args=(x, u, xcg, const, model, adjust_cy), method='Nelder-Mead', tol=minimize_tol, \
    #                options={'maxiter': maxiter})

    # cost = res.fun
    # #===================================================================================

    ##=== FMIN Algorithm ================================================================
    s = fmin(clf16, s, args=(x, u, xcg, const, model, adjust_cy), xtol=1e-12, maxiter=2000)

    J = clf16(s, x, u, xcg, const, model, adjust_cy)
    if printOn:
        print(f"cost = {J}")

    if orient != 3:
        x[1] = s[2]
        u[0] = s[0]
        u[1] = s[1]
    else:
        x[1] = s[0]
        u[0] = s[1]
        u[1] = s[2]
        u[2] = s[3]
        u[3] = s[4]
    

    ##===================================================================================

    if printOn:
        print(f'Throttle (percent):            {u[0]}')
        print(f'Elevator (deg):                {u[1]}')
        print(f'Ailerons (deg):                {u[2]}')
        print(f'Rudder (deg):                  {u[3]}')
        print(f'Angle of Attack (deg):         {rtod*x[1]}')
        print(f'Sideslip Angle (deg):          {rtod*x[2]}')
        print(f'Pitch Angle (deg):             {rtod*x[4]}')
        print(f'Bank Angle (deg):              {rtod*x[3]}')

        amach, qbar = adc(x[0], x[11])
        print(f'Dynamic Pressure (psf):        {qbar}')
        print(f'Mach Number:                   {amach}')

        # print('')
        # print(f'Cost Function:           {cost}')
    # assert cost < tol, "trimmerFun did not converge"

    return x, u
