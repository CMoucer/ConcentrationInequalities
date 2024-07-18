######################################################################################################################
# This file computes an upper bound to Hoeffding's inequality in the variational approach for i.i.d. random
# variables. It corresponds to Figure 2 of the paper.
######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from compute_concentration.compute_concentration_hoeffding import compute_hoeffding_exponential, compute_variational_hoeffding
from compute_concentration.compute_concentration_classical_bounds import compute_hoeffding_bound
from compute_concentration.compute_extremal_points_hoeffding import compute_extremal_points


n = 3
mu = 0.6
t_max = 1 - mu - 0.05
nb = 50
ts = np.linspace(0, t_max, nb)

bound_hoeffding = np.zeros(nb)
bound_variational = np.zeros(nb)
bound_exponential = np.zeros(nb)

for i in range(nb):
    bound_hoeffding[i] = compute_hoeffding_bound(ts[i], n)
    bound_exponential[i] = compute_hoeffding_exponential(mu, ts[i], n)
    extremal_points = compute_extremal_points(mu, ts[i], n)
    bound_variational[i] = compute_variational_hoeffding(extremal_points, mu)

plt.plot(ts, bound_hoeffding, color='green', label="Hoeffding\'s inequality")
plt.plot(ts, bound_exponential, color='blue', label='separable')
plt.plot(ts, bound_variational, color='red', label='variational')

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.xlabel('deviation t', fontsize=16)
plt.ylabel(r'concentration bound $\rho$', fontsize=16)
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()