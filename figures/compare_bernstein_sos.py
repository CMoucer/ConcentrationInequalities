######################################################################################################################
# This file computes an upper bound to Bernstein's inequality in the polynomial approach.
# It corresponds to Figure 5 of the paper.
######################################################################################################################


import numpy as np
import matplotlib.pyplot as plt

from compute_concentration_sos.compute_sos_bernstein import concentration_sos_bernstein
from compute_concentration.compute_concentration_classical_bounds import bernstein_inequality

mu_1 = -0.3 # first-order moment
mu_2 = 0.1 # second-order moment
c = 1 # X takes its values a.s. in [-c, c]
n = 2 # number of random variables
d = 2 # dimension of the SDP approximation

nb = 30
ts = np.linspace(0, 1 - mu_1 - 0.05, nb)

bernstein_bounds = np.zeros(nb)
sos_bounds = np.zeros(nb)

for i in range(nb):
    bernstein_bounds[i] = bernstein_inequality(mu_1, mu_2, ts[i], n, c)
    sos_bounds[i] = concentration_sos_bernstein(mu_1, mu_2, ts[i], d, n)

plt.plot(ts, bernstein_bounds, color='green', label='Bernstein')
plt.plot(ts, sos_bounds, color='red', label='SoS')

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.xlabel('deviation t', fontsize=16)
plt.ylabel(r'concentration bound $\rho$', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()