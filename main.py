######################################################################################################################
# This main.py provides a concentration bound in the separable and variational approaches, for n i.i.d. random variables
# with mean mu, compared to the classical Hoeffding's inequality.
######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from compute_concentration.compute_concentration_hoeffding import compute_hoeffding_exponential, compute_variational_hoeffding
from compute_concentration.compute_concentration_classical_bounds import compute_hoeffding_bound
from compute_concentration.compute_extremal_points_hoeffding import compute_extremal_points

n = 3 # number of i.i.d. random variables
mu = 0.6 # first-order moment
t = 0.1 # deviation (must be smaller than 1 - mu)

bound_hoeffding = compute_hoeffding_bound(t, n)
bound_exponential = compute_hoeffding_exponential(mu, t, n)
extremal_points = compute_extremal_points(mu, t, n)
bound_variational = compute_variational_hoeffding(extremal_points, mu)

print(f'Considering {n} i.i.d. random variables with a first-order moment {mu}, their average deviates from {t} with probability:')
print('Hoeffding\'s inquality: ', bound_hoeffding)
print('Separable approach: ', bound_exponential)
print('Variational approach:', bound_variational)
