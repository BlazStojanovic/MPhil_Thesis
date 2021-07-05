"""
Figure of DMC in 1D

"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

rng = default_rng(101010)

def V(x, k=1.0):
	return 0.5*k*k*np.square(x)

def weight(x, xp, dt, E0):
	return np.exp(-0.5*dt*(V(x) + V(xp)) + dt*E0)

def dmc(N, dt, steps, D, E0, xmin, xmax):
	X = np.zeros((N, steps))
	init = (xmax-xmin)*rng.uniform(size=N) + xmin # initial distr range
	X[:, 0] = init # initial distr range
	T = np.zeros((N, steps))
	walkers = np.zeros((steps,))
	walkers[0] = N

	trajectories = [] # this will be ugly, but here we will append finished trajectories + times
	times = [] # if the walks are displayed in black

	for i in range(1, steps):
		print("T = ", dt*i)
		# choose the random move from Gaussian with variance = 2*D*dt and mean = 0
		dX = rng.normal(loc=0.0, scale=2*D*dt, size=np.shape(X[:, i-1]))

		# move the configurations
		X[:, i] = X[:, i-1] + dX

		# Calculate weights 
		w = weight(X[:, i-1], X[:, i], dt, E0)

		# displace weights and turn into integer
		branch = (w + rng.random(np.shape(w))).astype(int)
		print("no. new walkers = ", np.sum(branch))
		print('-------------------------')
		Nnew = np.sum(branch) # number of walks in the next step
		walkers[i] = Nnew

		# cleanup the trajectories, branch and kill walkers
		s = 0 # no. deletes
		for j, b in enumerate(branch):
			if b == 0:
				# kill walker
				trajectories.append(X[j-s, 0:i+1])
				l = len(X[j-s, 0:i+1])
				T = dt*(i*np.ones(l)-np.arange(l)[::-1])
				times.append(T) # current time is the last time in the trajectory! t = dt to here
				X = np.delete(X, j-s, axis=0)
				s += 1
			elif b > 1:
				# branch walker into b new ones
				X = np.insert(X, j-s, np.tile(X[j-s, :], (b-1, 1)) , axis=0)

	time = np.arange(steps)*dt
	return X, trajectories, times, time, init, walkers

if __name__ == '__main__':
	# params of the simulation
	N = 1000 # initial number of walkers
	dt = 0.05 # time step
	steps = 500 # no. time steps
	D = 0.5 # diffusion constant hbar = 1, me = 1
	E0 = 0.25

	# initial ensemble
	xmin = -2
	xmax = +2

	# X, trajectories, times, time, init, walkers = dmc(N, dt, steps, D, E0, xmin, xmax)
	
	# # save stuff 
	# np.save("figdata/dmc_killed_trajects1.npy", trajectories)
	# np.save("figdata/dmc_trajects1.npy", X)
	# np.save("figdata/dmc_times1.npy", times)
	# np.save("figdata/dmc_time1.npy", time)
	# np.save("figdata/dmc_init1.npy", init)
	# np.save("figdata/dmc_walkers1.npy", walkers)

	# second round for plotting
	N = 7
	X, trajectories, times, time, init, walkers = dmc(N, dt, steps, D, E0, xmin, xmax)
	
	# save stuff 
	np.save("figdata/dmc_killed_trajects2.npy", trajectories)
	np.save("figdata/dmc_trajects2.npy", X)
	np.save("figdata/dmc_times2.npy", times)
	np.save("figdata/dmc_time2.npy", time)
	np.save("figdata/dmc_init2.npy", init)
	np.save("figdata/dmc_walkers2.npy", walkers)


