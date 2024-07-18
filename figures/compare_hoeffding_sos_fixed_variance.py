######################################################################################################################
# This file computes an upper bound to Hoeffding's inequality in the polynomial approach for i.i.d. random variables,
# with a fixed second-order moment. It corresponds to Figure 7 of the paper.
######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from compute_concentration.compute_concentration_classical_bounds import compute_hoeffding_bound
from compute_concentration_sos.compute_sos_hoeffding import concentration_sos_hoeffding_with_variance
from compute_concentration.compute_concentration_hoeffding import compute_variational_hoeffding
from compute_concentration.compute_extremal_points_hoeffding import compute_extremal_points

mu_1 = 0.3 # first-order moment
mu_2 = 0.1 # second-order moment
n = 2 # number of random variables
d = 2 # SoS approximation

nb = 30
ts = np.linspace(0, 1 - mu_1 - 0.05, nb)

hoeffding_bounds = np.zeros(nb)
sos_bounds = np.zeros(nb)
linear_bounds = np.zeros(nb)
variational_bounds = np.zeros(nb)
exponential_bounds = np.zeros(nb)

computed = False

for i in range(nb):
    hoeffding_bounds[i] = compute_hoeffding_bound(ts[i], n)
    linear_bounds[i] = mu_1 / (mu_1 + ts[i])
    extremal_points = compute_extremal_points(mu_1, ts[i], n)
    variational_bounds[i] = compute_variational_hoeffding(extremal_points, mu_1)
if computed:
    sos_bounds = np.genfromtxt(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/figures/data/hoeffding_n_{n}_d_{d}_mu1_{mu_1}_mu2_{mu_2}.txt',
        delimiter=',')
else:
    for i in range(nb):
        print(ts[i])
        sos_bounds[i] = concentration_sos_hoeffding_with_variance(mu_1, mu_2, ts[i], d, n)
    np.savetxt(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/figures/data/hoeffding_n_{n}_d_{d}_mu1_{mu_1}_mu2_{mu_2}.txt',
        sos_bounds, delimiter=',')

plt.plot(ts, hoeffding_bounds, color='black', label='hoeffding')
plt.plot(ts, sos_bounds, color='red', label='SoS')
plt.plot(ts, variational_bounds, '--', color='blue', label='variational')
plt.plot(ts, mu_1/(mu_1 + ts), '--', color='green', label='linear')

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.xlabel('deviation t', fontsize=16)
plt.ylabel(r'concentration bound $\rho$', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.savefig(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/output/hoeffding/comparison_sos_mu1_{mu_1}_mu_2_{mu_2}_d_{d}_n_{n}.png', dpi=150)
plt.show()
