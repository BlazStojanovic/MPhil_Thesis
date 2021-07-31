import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

def ReLu(x):
	if x > 0:
		return x
	else:
		return 0

def Softplus(x):
	return np.log(1 + np.exp(x))

x = np.linspace(-3, 3, 1000)
y1 = [ReLu(a) for a in x]
y2 = Softplus(x)

fig = plt.figure(figsize=(12, 7))
plt.plot(x, y1, label='ReLu', color='blue', linewidth=2)
plt.plot(x, y2, label='Softplus', color='red', linewidth=2)

plt.xlabel("$x$")
plt.ylabel(r"$g(x)$")
plt.legend()
plt.savefig("../Chapter4/Figs/Vector/activations.pdf", bbox_inches='tight')
# plt.show()

