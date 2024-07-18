######################################################################################################################
# This file computes an upper bound to Bennett's inequality in the polynomial approach.
# It corresponds to Figure 6 of the paper.
######################################################################################################################


import numpy as np
import matplotlib.pyplot as plt

from compute_concentration_sos.compute_sos_bennett import concentration_sos_bennett, concentration_bennett_sos_complexified
from compute_concentration.compute_concentration_classical_bounds import bennett_inequality

mu_1 = -0.3 # first-order moment
mu_2 = 1 # second-order moment
a = 1 # maximal value for X
n = 2 # number of random variables
d = 2 # SoS approximation
sigma = np.sqrt(n * (mu_2 - mu_1 **2)) # global variance for i.i.d. random variables

nb = 30
ts = np.linspace(0.05, 1 - mu_1 - 0.05, nb)

bennet_bounds = np.zeros(nb)
sos_bounds = np.zeros(nb)


for i in range(nb):
    bennet_bounds[i] = bennett_inequality(sigma, a, ts[i], n)
    sos_bounds[i] = concentration_bennett_sos_complexified(mu_1, mu_2, ts[i], d, n)

plt.plot(ts, bennet_bounds, color='green', label='Bennett')
plt.plot(ts, sos_bounds, color='red', label='SoS')

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.xlabel('deviation t', fontsize=16)
plt.ylabel(r'concentration bound $\rho$', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()