import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import matplotlib.gridspec as gridspec


matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def V(x, k=1.0):
	return 0.5*k*k*np.square(x)

def unif(x, m):
	if np.abs(x) < m:
		return 1.0/(2*m)
	else:
		return 0.0

unif = np.vectorize(unif)

def HO(x):
	return 1/np.power(np.pi, 0.25)*np.exp(-np.square(x)/2) 

trajectories = np.load("figdata/dmc_killed_trajects1.npy", allow_pickle=True)
X = np.load("figdata/dmc_trajects1.npy")
times = np.load("figdata/dmc_times1.npy", allow_pickle=True)
time = np.load("figdata/dmc_time1.npy")
init = np.load("figdata/dmc_init1.npy")
walkers = np.load("figdata/dmc_walkers1.npy")

trajectories1 = np.load("figdata/dmc_killed_trajects2.npy", allow_pickle=True)
X1 = np.load("figdata/dmc_trajects2.npy")
times1 = np.load("figdata/dmc_times2.npy", allow_pickle=True)
time1 = np.load("figdata/dmc_time2.npy")

fig, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [1, 6, 1], 'height_ratios':[2, 7]})
fig.set_size_inches(14, 8)

ax[1, 2].sharey(ax[1, 1])
ax[1, 2].set_yticks([])
ax[1, 1].set_xlim([0, 25])

ax[0, 1].sharex(ax[1, 1])
ax[0, 1].xaxis.set_label_position('top') 
ax[0, 1].xaxis.tick_top()
ax[0, 1].set_xlabel('t')
ax[0, 0].axis('off')
ax[0, 2].axis('off')

# plot potential
tt = np.linspace(0, 25, 2)
xx = np.linspace(-3, 3, 1000)
TT, XX = np.meshgrid(tt, xx)
img = V(XX)
ax[1,1].pcolormesh(TT, XX, img, cmap='Blues', alpha=1.0)

ax[1,1].text(20, 2.3, r"$V(x) = \frac{1}{2}x^2$")

# plot no. walkers
ax[0, 1].plot(time, walkers, color='black', linewidth=2)
ax[0, 1].fill_between(time, walkers, color='grey', linewidth=2, alpha=0.5)
ax[0, 1].set_ylabel(r'$N_{w}$')
ax[0, 1].ticklabel_format(style='sci', axis='y')
ax[0, 1].set_ylim([0, 1100])

# first plots
for i, x in enumerate(X):
	ax[1, 1].plot(time, x, color='grey', alpha=0.1)

for i, x in enumerate(trajectories):
	ax[1, 1].plot(times[i], x, color='grey', alpha=0.1)

# second plots
for i, x in enumerate(X1):
	ax[1, 1].plot(time, x, color='black')

for i, x in enumerate(trajectories1):
	ax[1, 1].plot(times1[i], x, color='black')

# plot end histogram and analytical solution
x = np.linspace(-3, 3, 1000)
ax[1, 2].hist(X[:, -1], alpha=0.8, orientation='horizontal', color='blue', density=True, bins=20, edgecolor='black')
ax[1, 2].plot(np.square(HO(x)), x, '--', color='black', linewidth=2, label=r'$|u_0|^2 = \frac{1}{\pi^{1/2}}e^{-x^2}$')
# ax[1, 2].legend()

# plot start histogram and analytical solution
ax[1, 0].hist(init, alpha=0.8, orientation='horizontal', color='blue', density=True, bins=20, edgecolor='black')
ax[1, 0].plot(unif(x, 2), x, '--', color='black', linewidth=2)

ax[1, 1].set_xlabel('t')
ax[1, 0].set_ylabel('x')
ax[1, 0].set_xlabel(r'$|\Psi(t=0)|^2$')
ax[1, 2].set_xlabel(r'$|\Psi_0|^2$')
ax[1, 2].set_xlim([0, 0.6])
ax[1, 0].set_xlim([0, 0.6])

plt.subplots_adjust(wspace=0.05, hspace=0.05)

plt.savefig('../Chapter2/Figs/Raster/dmc.png', bbox_inches='tight')
plt.show()