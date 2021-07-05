import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

# Brownian motion
def bmotion(t=1, n=1001, N=30):
	dt = t/(n-1)
	t = np.linspace(0,t,n)

	dX = np.sqrt(dt) * np.random.randn(N, n)
	X = np.cumsum(dX, axis=1)
	return X, t

def V(x):
	# return np.power(x, 2)
	return x

def FKmeasure(x, dt):
	# simple euler scheme, TODO check validity!
	v = V(x)
	return np.exp(-np.sum(v, axis=1)*dt)


np.random.seed(42069)
N = 30 # number of brownian paths
t = 1.0 # time length
n = 1001 # number of steps
dt = t/(n-1)

X, t = bmotion(t=t, n=n, N=N)

## First figure, N amount of Brownian processes with different initial starting conditions
# fig = plt.figure(figsize=(7, 5))

# for i in range(N):
#     plt.plot(t, X[i,:], color='black', alpha=0.7, zorder=1)

# plt.text(0.05, 2, '$V(x) = 0$')

# plt.xlabel('$t$')
# plt.ylabel('$X_t$')
# plt.xlim([0,1])
# plt.ylim([-2.5, 2.5])
# plt.savefig("../Chapter2/Figs/Raster/reweight1.pdf", bbox_inches='tight')
# plt.show()

measure = FKmeasure(X, dt)
print(measure)
measure = measure/np.max(measure)
print(measure)

# Background for second plot
tt = np.linspace(0, 1, 2)
xx = np.linspace(-2.5, 2.5, 1000)

TT, XX = np.meshgrid(tt, xx)
img = V(XX)

## Second figure, now with potential
fig = plt.figure(figsize=(7, 5))

colors = plt.cm.binary(measure)

for i in range(N):
    plt.plot(t, X[i,:], color=colors[i], zorder=3)

plt.text(0.05, 2, '$V(x) = x$')

plt.pcolormesh(TT, XX, img, cmap='Blues_r', alpha=1.0)
plt.xlabel('$t$')
plt.ylabel('$X_t$')
plt.xlim([0,1])
plt.ylim([-2.5, 2.5])
plt.savefig("../Chapter3/Figs/Raster/reweight2.png", bbox_inches='tight', dpi=120)
plt.show()
