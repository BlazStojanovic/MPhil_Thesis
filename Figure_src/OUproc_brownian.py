import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

np.random.seed(42070)

# Brownian motion
def bmotion(t=1, n=1001, N=30):
	dt = t/(n-1)
	t = np.linspace(0,t,n)

	dX = np.sqrt(dt) * np.random.randn(N, n)
	X = np.cumsum(dX, axis=1)
	return X, t


## First subfigure, zoom in on the Wiener process
# fig, ax = plt.subplots(figsize=[7, 5])
# axins = ax.inset_axes([0.5,0.1,0.45,0.45])
# xi1, xi2, yi1, yi2 = 0.255, 0.314, 0.34, 0.55
# axins.set_xlim(xi1, xi2)
# axins.set_ylim(yi1, yi2)

# X, t = bmotion(N=1, n=100001)

# ax.set_xlabel('$t$')
# ax.set_ylabel('$X_t$')

# ax.plot(t, X.T, color='black')
# axins.plot(t, X.T, color='black')
# axins.set_xticks([])
# axins.set_yticks([])
# ax.indicate_inset_zoom(axins, edgecolor='black')

# plt.savefig("../Chapter3/Figs/Raster/OU-BM-I.png", bbox_inches='tight')
# plt.show()


# Ohrenstein-Uhlenbeck process
np.random.seed(1)


def OU_avg(t, theta, mu, x0):
	return x0*np.exp(-theta*t) + mu*(1-np.exp(-theta*t))


def OU_var(t, theta, mu, sigma):
	return sigma**2/(2*theta)*(1-np.exp(-2*theta*t))

def dW(dt):
    return np.random.normal(loc=0.0, scale=np.sqrt(dt))

def OU_samples(dt, t, theta, mu, x0, sigma, N):
	# with Euler-Maruyama
	X = np.zeros((N, len(t)))
	X[:, 0] = x0 # set initial state

	for i in range(N):
		for j in range(1, len(t)):
			x = X[i, j-1]
			X[i, j] = x+theta*(mu-x)*dt+sigma*dW(dt)

	return X

theta = 0.6
sigma = 1.1
x0 = 1.0
mu = -1.3
t = np.linspace(0, 5, 10001)
dt = 5./10000.

Xt = OU_samples(dt, t, theta, mu, x0, sigma, 10)


fig, ax = plt.subplots(figsize=[7, 5])

# Plot sample paths
ax.plot(t, Xt.T, color='k', linewidth=0.5, alpha=0.5)

# Plot analytical solutions
ax.plot(t, OU_avg(t, theta, mu, x0), color='blue', label='$m(t)$', linewidth=3)
ax.fill_between(t, OU_avg(t, theta, mu, x0)+2*OU_var(t, theta, mu, sigma), OU_avg(t, theta, mu, x0)-2*OU_var(t, theta, mu, sigma),  color='blue', alpha=0.1, label=r'$\pm 2\mathrm{Var}[X_t]$')
ax.legend()

ax.plot(t, OU_avg(t, theta, mu, x0)+2*OU_var(t, theta, mu, sigma), '--', color='blue', linewidth=3)
ax.plot(t, OU_avg(t, theta, mu, x0)-2*OU_var(t, theta, mu, sigma), '--', color='blue', linewidth=3)

ax.set_xlabel("$t$")
ax.set_ylabel("$X_t$")

plt.savefig("../Chapter3/Figs/Raster/OU-BM-II.png", bbox_inches='tight')
plt.show()