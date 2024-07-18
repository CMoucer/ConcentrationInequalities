import numpy as np
import matplotlib.pyplot as plt

from compute_concentration_sos.compute_sos_bennet import concentration_sos_bennet, concentration_bennet_sos_complexified
from compute_concentration.compute_concentration_classical_bounds import bennet_inequality

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

computed = False

for i in range(nb):
    bennet_bounds[i] = bennet_inequality(sigma, a, ts[i], n)
if computed:
    sos_bounds = np.genfromtxt(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/figures/data/bennet_n_{n}_d_{d}_mu1_{mu_1}_mu2_{mu_2}.txt',
        delimiter=',')
else:
    for i in range(nb):
        print(ts[i])
        #sos_bounds[i] = concentration_sos_bennet(mu_1, mu_2, ts[i], d, n)
        sos_bounds[i] = concentration_bennet_sos_complexified(mu_1, mu_2, ts[i], d, n)
    np.savetxt(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/figures/data/bennet_n_{n}_d_{d}_mu1_{mu_1}_mu2_{mu_2}.txt',
        sos_bounds, delimiter=',')

plt.plot(ts, bennet_bounds, color='green', label='Bennett')
plt.plot(ts, sos_bounds, color='red', label='SoS')

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.xlabel('deviation t', fontsize=16)
plt.ylabel(r'concentration bound $\rho$', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.savefig(f'/Users/cmoucer/PycharmProjects/ConcentrationInequalities_ConvOpt/output/bennet/comparison_sos_mu1_{mu_1}_mu_2_{mu_2}_d_{d}_n_{n}.png', dpi=150)
plt.show()