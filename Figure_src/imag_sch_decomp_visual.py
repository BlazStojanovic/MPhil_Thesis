import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

Ens = np.arange(1, 4)*np.array([0.5, 1, 2])
x = np.linspace(0, 3, 1000)
c = ['red', 'blue', 'green']
f = lambda x, E: np.exp(-E*x)

# Exponential decays
for i, En in enumerate(Ens):
	plt.figure(figsize=(4, 4))
	plt.text(1.2, 0.8, "$E_{} = {}$".format(i, En))
	plt.plot(x, f(x, En), color=c[i], linewidth=4)
	# plt.xlabel("t")
	plt.xticks([])
	plt.yticks([])
	plt.savefig("../../Midterm_assessment/figures/ImgSch-{}.png".format(i), bbox_inches='tight')
	plt.close()