import numpy as np
import cvxpy as cp


def chebyshev_inequality_exact(mu, sigma, t):
    beta_1 = cp.Variable(1)
    beta_2 = cp.Variable(1)
    alpha = cp.Variable(1)
    tau = cp.Variable(1)
    S = cp.Variable((2, 2))

    A = np.zeros((2, 2))
    A[0][1] = 1/2
    A[1][0] = 1/2
    A[1][1] = -(mu + t)
    #print(A)
    B = np.zeros((2, 2))
    B[1][1] = 1
    #print(B)

    constraints = [tau >= 0]
    constraints += [S[0][0] == beta_2[0]]
    constraints += [S[1][0] == beta_1[0]/2]
    constraints += [S[0][1] == beta_1[0]/2]
    constraints += [S[1][1] == alpha[0]]
    constraints += [S - B - tau * A >> 0]
    constraints += [S >> 0]

    objective = cp.Minimize(beta_2 * sigma ** 2 + beta_1 * mu + alpha)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)
    return prob.value