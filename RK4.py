from numpy import array

def RK4(f, time, dt, xx, u, xcg):
	"""
	fourth-order Runge-Kutta Algorithm
	"""
	# k1
	xd = f(xx, u, Xcg=xcg)[0]
	xa = xd*dt
	# k2
	x = xx + 0.5*xa
	t = time + 0.5*dt
	xd = f(x, u, Xcg=xcg)[0]
	q = xd*dt
	# k3
	x = xx + 0.5*q
	xa = xa + 2.0*q
	xd = f(x, u, Xcg=xcg)[0]
	q = xd*dt
	# k4
	x = xx + q
	xa = xa + 2.0*q
	time = time + dt
	xd = f(x, u, Xcg=xcg)[0]
	
	# x_new	
	xnew = xx + (xa + xd*dt)/6.0
	return xnew

