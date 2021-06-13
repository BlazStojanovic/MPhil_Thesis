import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

# Gaussian 
def gaussian(X, mu=0.0, sig=1.0):
	return 1/sig/np.sqrt(2*np.pi)*np.exp(-0.5*np.square(X-mu)/np.square(sig))

# Brownian motion
def bmotion(v, xsmin, xsmax, t=1, n=1001, N=30):
	dt = t/(n-1)
	t = np.linspace(0,t,n)
	x0 = np.zeros((N, 1))
	x0[:, 0] = np.linspace(xsmin, xsmax, N, endpoint=True)


	dX = np.sqrt(dt) * np.random.randn(N, n)
	X = np.cumsum(dX, axis=1) + x0
	return X, t

np.random.seed(42069)
N = 10000 # number of brownian paths
v = 0.0 # potential
t = 1.0 # time length
n = 1001 # number of steps
dt = t/(n-1)

t1 = 0.1 # query time 1
t2 = 0.4 # query time 2
t3 = 0.9 # query time 3

X, t = bmotion(v, -2, 2, N=N) # generate Brownian walks

# # First figure, N amount of Brownian processes with different initial starting conditions
# fig = plt.figure(figsize=(14, 5))

# colors = plt.cm.coolwarm(np.linspace(0,1,N))

# plt.vlines([t1, t2, t3], -3.5, 3.5, color='black', linewidth=2, zorder=2) # time query lines
# plt.text(t1+0.01, 3, '$t_1$')
# plt.text(t2+0.01, 3, '$t_2$')
# plt.text(t3+0.01, 3, '$t_3$')

# for i in range(N):
#     plt.plot(t, X[i,:], color=colors[i], alpha=0.7, zorder=1)


# plt.xlabel('$t$')
# plt.ylabel('$X_t$')
# plt.xlim([0,1])
# plt.ylim([-3.5, 3.5])
# plt.savefig("../Chapter3/Figs/Raster/fkac_vs_fplanck_top.pdf", bbox_inches='tight')
# plt.show()
# plt.close()

# """Second figure, at query point t=t2, and illustration of random walks passing through query positions between
# (x11, x12) and (x21, x22), plot of wave function will be betweeen (-2, 2), thus for a histogram with ~80 
# bars, we will have width of gate 0.05"""
# x11 = 1.50 # gate 1
# x12 = 1.60
# x21 = -0.40 # gate 2
# x22 = -0.30

# # filter the random walks into the three categories
# m2 = int(t2/dt) # index of t1 checkpoint
# mask1 = np.logical_and((x11 < X[:, m2]), (X[:, m2] < x12))
# mask2 = np.logical_and((x21 < X[:, m2]), (X[:, m2] < x22))

# gate1_walks = X[mask1]
# gate2_walks = X[mask2]
# non_walks = X[np.logical_not(np.logical_or(mask1, mask2))]

# fig, ax = plt.subplots(figsize=[14, 5])
# axins = ax.inset_axes([0.6,0.05,0.35,0.75])
# xi1, xi2, yi1, yi2 = t2-0.01, t2+0.01, 1.2, 1.9
# axins.set_xlim(xi1, xi2)
# axins.set_ylim(yi1, yi2)

# # plot the rest of the walks
# for i in range(np.shape(non_walks)[0]):
#     ax.plot(t, non_walks[i,:], color='grey', alpha=0.2, linewidth=0.5, zorder=1)
#     axins.plot(t, non_walks[i,:], color='grey', alpha=0.2, linewidth=0.5, zorder=1)

# # plot the walks going through gate1
# for i in range(np.shape(gate1_walks)[0]):
#     ax.plot(t, gate1_walks[i,:], color=plt.cm.coolwarm(0.1), alpha=1.0, linewidth=2, zorder=3)
#     axins.plot(t, gate1_walks[i,:], color=plt.cm.coolwarm(0.1), alpha=1.0, linewidth=2, zorder=3)

# # plot the walks going through gate2
# for i in range(np.shape(gate2_walks)[0]):
#     ax.plot(t, gate2_walks[i,:], color=plt.cm.coolwarm(0.9), alpha=1.0, linewidth=2, zorder=2)
#     axins.plot(t, gate2_walks[i,:], color=plt.cm.coolwarm(0.9), alpha=1.0, linewidth=2, zorder=2)


# ax.vlines([t2], -3.5, 3.5, color='black', linewidth=2, zorder=4)
# axins.vlines([t2], -3.5, 3.5, color='black', linewidth=2, zorder=4)
# ax.text(t2+0.01, 3, '$t_2$', zorder=10)
# axins.text(t2+0.001, x12+0.1, '$t_2$', zorder=10)

# # mark gates on first plot
# ax.scatter(t2, 0.5*(x11 + x12), color='black', s=150, zorder=5, marker='o')
# ax.scatter(t2, 0.5*(x21 + x22), color='black', s=150, zorder=5, marker='s')

# axins.scatter(t2, x11, color='black', s=150, zorder=5, marker=6)
# axins.scatter(t2, x12, color='black', s=150, zorder=5, marker=7)

# ax.set_xlabel('$t$')
# ax.set_ylabel('$X_t$')
# ax.set_xlim([0,1])
# ax.set_ylim([-3.5, 3.5])

# axins.set_xlabel('$t$')
# axins.set_ylabel('$X_t$')
# ax.indicate_inset_zoom(axins, edgecolor='black')
# axins.xaxis.tick_top()
# axins.xaxis.set_label_position('top') 

# plt.savefig("../Chapter3/Figs/Raster/fkac_vs_fplanck_mid1.pdf", bbox_inches='tight')
# plt.show()
# plt.close()

# # Figure 3: Paths reweighted on the basis of a potential
# def V(x):
# 	return np.square(0.5-x)

# Figure 4: F-Kac solution to the problem from the generated paths
dx = 0.1
xmin = -2.2
xmax =  2.2

def FKac(X, dt, t, w, xmin, xmax, dx):
	m = int(t/dt)
	Nsamp = int((xmax-xmin)/dx)
	psi = np.zeros((Nsamp, 1))

	# for each gate
	for i in range(Nsamp):
		# find appropriate walks
		x1 = xmin+i*dx
		x2 = xmin+(i+1)*dx
		walks = np.logical_and((x1 < X[:, m]), (X[:, m] < x2))

		# weighted average over walks
		psi[i] = np.average(gaussian((X[walks])[:, 0]))

	return psi

N = 10000 # bigger N for nicer results
fig = plt.figure(figsize=(14, 5))
x = np.linspace(-2, 2, 1000)

# plot initial distribution
plt.plot(x, gaussian(x), 'black', linewidth=3, label=r"$\psi_0$")

# plot solution at t1
xs = xmin + np.arange(int((xmax-xmin)/dx))*dx
psi_t1 = FKac(X, dt, t1, 1.0, xmin, xmax, dx)
plt.step(xs, psi_t1, where='mid', color='blue', linewidth=2, label=r"$\psi(x, t_1={})$".format(t1))

# plot solution at t2
xs = xmin + np.arange(int((xmax-xmin)/dx))*dx
psi_t2 = FKac(X, dt, t2, 1.0, xmin, xmax, dx)
plt.step(xs, psi_t2, where='mid', color='red', linewidth=2, label=r"$\psi(x, t_2={})$".format(t2))

# plot solution at t3
xs = xmin + np.arange(int((xmax-xmin)/dx))*dx
psi_t3 = FKac(X, dt, t3, 1.0, xmin, xmax, dx)
plt.step(xs, psi_t3, where='mid', color='green', linewidth=2, label=r"$\psi(x, t_3={})$".format(t3))

plt.legend()
plt.ylabel("$\psi(x, t)$")
plt.xlabel("$x$")
plt.savefig("../Chapter3/Figs/Raster/fkac_vs_fplanck_bottom.pdf", bbox_inches='tight')
plt.show()