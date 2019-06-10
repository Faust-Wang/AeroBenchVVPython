'''
Stanley Bak
Python F-16

Apply constraints to x variable
used when finding trim conditions
'''

from math import sin, cos, asin, sqrt, tan, atan
from tgear import tgear

## coordinate turn constraints
def turn_coord_cons(turn_rate, alpha, beta, TAS, gamma=0):
    """Calculates phi for coordinated turn.
    """

    g0 = 32.17
    G = turn_rate * TAS / g0

    if abs(gamma) < 1e-8:
        phi = G * cos(beta) / (cos(alpha) - G * sin(alpha) * sin(beta))
        phi = atan(phi)
    else:
        a = 1 - G * tan(alpha) * sin(beta)
        b = sin(gamma) / cos(beta)
        c = 1 + G ** 2 * cos(beta) ** 2

        sq = sqrt(c * (1 - b ** 2) + G ** 2 * sin(beta) ** 2)

        num = (a - b ** 2) + b * tan(alpha) * sq
        den = a ** 2 - b ** 2 * (1 + c * tan(alpha) ** 2)

        phi = atan(G * cos(beta) / cos(alpha) * num / den)
    return phi

## ROC constraints
def rate_of_climb_cons(gamma, alpha, beta, phi):
    """Calculates theta for the given ROC, wind angles, and roll angle.
    """
    a = cos(alpha) * cos(beta)
    b = sin(phi) * sin(beta) + cos(phi) * sin(alpha) * cos(beta)
    sq = sqrt(a ** 2 - sin(gamma) ** 2 + b ** 2)
    theta = (a * b + sin(gamma) * sq) / (a ** 2 - sin(gamma) ** 2)
    theta = atan(theta)
    return theta


## constraints
def conf16(x, u, const):
    'apply constraints to x'

    radgam, singam, rr, pr, tr, phi, cphi, sphi, thetadot, coord, stab, orient = const
    gamm = asin(singam)

    #
    # Steady Level Flight
    #
    if orient == 1:
        x[3] = phi          # Phi
        x[4] = x[1]         # Theta
        x[6] = rr           # Roll Rate
        x[7] = pr           # Pitch Rate
        x[8] = 0.0          # Yaw Rate

    #
    # Steady Climb
    #
    if orient == 2:
        x[3] = phi          # Phi
        x[4] = x[1] + radgam  # Theta
        x[6] = rr           # Roll Rate
        x[7] = pr           # Pitch Rate
        x[8] = 0.0          # Yaw Rate

    #
    # orient=3 implies coordinated turn
    #
    if orient == 3: # tr = turn rate (page 342 book)
        x[3] = phi # turn_coord_cons(tr, x[1], x[2], x[0], gamma=gamm)
        x[4] = rate_of_climb_cons(gamm, x[1], x[2], phi) # theta
        x[6] = -tr * sin(x[4])            # Roll Rate
        x[7] = tr * cos(x[4]) * sin(x[3])  # Pitch Rate
        x[8] = tr * cos(x[4]) * cos(x[3])  # Yaw Rate

    #
    # Pitch Pull Up
    #
    if orient == 4:
        x[4] = x[1]         # Theta = alpha
        x[3] = phi          # Phi
        x[6] = rr           # Roll Rate
        x[7] = thetadot     # Pitch Rate
        x[8] = 0.0          # Yaw Rate

    x[12] = tgear(u[0])

    return x, u
