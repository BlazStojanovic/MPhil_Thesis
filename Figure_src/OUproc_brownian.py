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


## First subfigure, zoom in on the Wiener process
fig, ax = plt.subplots(figsize=[7, 5])
axins = ax.inset_axes([0.6,0.05,0.35,0.75])
xi1, xi2, yi1, yi2 = 0, 0.1, 0, 0.1
axins.set_xlim(xi1, xi2)
axins.set_ylim(yi1, yi2)