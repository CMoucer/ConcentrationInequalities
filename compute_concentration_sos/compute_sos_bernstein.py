import numpy as np
import cvxpy as cp

from compute_concentration_sos.utils import monomial_vector, decomposition_SDP_to_monomials, kappas, new_kappas

def polytope_bernstein(mu, t, matrix_kappas):
    """
    This function returns the construction of the polytope for Bernstein's inequality in the kappas basis.
    :param mu: first-order moment (float)
    :param t: deviation parameter (float)
    :param matrix_kappas: decomposition of the matrix_kappas (np.ndarray)
    :return: (np.ndarray)
    """
    dim_sdp = matrix_kappas.shape[0]
    n = matrix_kappas.shape[2] # number of variables
    d = np.max(matrix_kappas) / 2 # degree of the polynomials
    G = np.zeros((2 * n + 1, dim_sdp, dim_sdp))
    for k in range(n):
        for i in range(dim_sdp):
            for j in range(dim_sdp):
                if np.sum(matrix_kappas[i][j]) == 1 and (matrix_kappas[i][j][k] == 1):
                    G[2 * k][i][j] = 1/2
                    G[2 * k + 1][i][j] = -1/2
                    G[2 * n][i][j] = 1/2
                elif np.sum(matrix_kappas[i][j]) == 0:
                    G[2 * k][i][j] = 1
                    G[2 * k + 1][i][j] = 1
                    G[2 * n][i][j] = - n * (mu + t)
    return G

def concentration_sos_bernstein(mu_1, mu_2, t, d_eff, n, verbose=False):
    """
    Compute the concentration for Bernstein's inequality, with a quadratic upper upper bound approximated
    by polynomials of degree d and their relaxed SDP .
    :param mu_1: first-order moment (float)
    :param mu_2: second-order moment (float)
    :param t: deviation to the mean (float)
    :param n: number of variables (int)
    :param d_eff: dimension of the polynomial approximation (int)
    :param verbose:
    :return:
    """
    print(f'The effective degree will be {d_eff + 1} due to the constraints on the bernstein polytop')
    d = d_eff + 1
    ## VARIABLES
    # Create the right variables
    monomials = np.array(monomial_vector(d, n))
    matrix_kappas, antecedent_kappas, value_kappas = kappas(monomials)
    # Dimensions and values of the problem
    dim_kappas = value_kappas.shape[0]
    dim_sdp = matrix_kappas.shape[0]
    # Definition of the variables
    Q = cp.Variable((dim_sdp, dim_sdp), symmetric=True)
    S = []
    X = []
    # Build the kappas representation for G
    G = polytope_bernstein(mu_1, t, matrix_kappas)
    G_kappas = np.zeros((G.shape[0], len(value_kappas)))
    for i in range(G_kappas.shape[0]):
        G_kappas[i] = decomposition_SDP_to_monomials(G[i], antecedent_kappas)
    nb_S = G.shape[0] + 1
    for i in range(nb_S):
        S.append(cp.Variable((dim_sdp, dim_sdp), symmetric=True))
        if i < nb_S - 1:
            X.append(cp.Variable((dim_sdp, dim_sdp), symmetric=True))
    # build the kappas representation of S, X, and Q
    S_kappas = [[] for l in range(nb_S)]
    X_kappas = [[] for l in range(nb_S - 1)]
    Q_kappas = []
    for l in range(nb_S):
        for k in antecedent_kappas.keys():
            idx = antecedent_kappas[k]
            S_kappas[l].append(sum(S[l][id[0], id[1]] for id in idx))
            if l < nb_S - 1:
                X_kappas[l].append(sum(X[l][id[0], id[1]] for id in idx))
    for k in antecedent_kappas.keys():
        idx = antecedent_kappas[k]
        Q_kappas.append(sum(Q[id[0], id[1]] for id in idx))
    assert len(S_kappas[0]) == dim_kappas

    ## CONSTRAINTS
    # Add psd constraints
    constraints = []
    for l in range(nb_S):
        constraints += [S[l] >> 0]
        if l < nb_S - 1:
            constraints += [X[l] >> 0]
    ### constraints on Q : representable by monomials (1, x_1, x_2)
    for i in range(dim_sdp):
        if sum(monomials[i]) >= 2:  # Q has a size of 3x3
            constraints += [Q[i, :] == 0]
            constraints += [Q[:, i] == 0]
    ### constraints on the degree of X_1, ..., X_l and S_1, ..., S_l EXCEPT ON S_0 and X_0
    for l in range(1, nb_S):
        for i in range(dim_sdp):
            for j in range(dim_sdp):
                if np.sum(matrix_kappas[i][j]) > 2 * d_eff:
                    constraints += [S[l][i][j] == 0]
                    if l < nb_S - 1:
                        constraints += [X[l][i][j] == 0]
    # add constraints on S_kappas and X_kappas
    for k in range(dim_kappas):
        sum_S = S_kappas[0][k]
        sum_X = X_kappas[0][k]
        for l in range(1, nb_S):
            for i in range(dim_kappas):
                for j in range(dim_kappas):
                    new_k = value_kappas[i] + value_kappas[j]
                    if (new_k == value_kappas[k]).all():
                        sum_S += S_kappas[l][i] * G_kappas[l-1][j]
                        if l < nb_S - 1:
                            sum_X += X_kappas[l][i] * G_kappas[l-1][j]
        if np.sum(value_kappas[k]) == 0:  # add the one inside the set
            constraints += [Q_kappas[k] - 1 == sum_S]
        else:
            constraints += [Q_kappas[k] == sum_S]
        constraints += [Q_kappas[k] == sum_X]

    ## DEFINITION OF THE OPTIMIZATION PROBLEM
    ### Build the mu matrix
    mus = np.zeros((len(monomials), len(monomials)))
    for i in range(len(monomials)):
        for j in range(len(monomials)):
            order_sum = np.sum(matrix_kappas[i][j])
            if np.sum(monomials[i]) <= 1 and np.sum(monomials[j]) <= 1:
                if order_sum == 0:
                    mus[i, j] = 1
                elif order_sum == 1:
                    mus[i][j] = mu_1
                elif order_sum == 2 and 1 in matrix_kappas[i][j]:
                    mus[i][j] = mu_1 ** 2
                else:
                    mus[i][j] = mu_2
    ### OBJECTIVE AND OPTIMIZATION PROBLEM
    objective = cp.Minimize(cp.trace(mus @ Q))
    # Define the problem
    problem = cp.Problem(objective, constraints)

    if verbose:
        problem.solve(solver=cp.SDPA, verbose=verbose)  # , maxIteration=100, epsilonDash=1.0E-10, mpfPrecision=250)
        print('solution to the problem', problem.value)
        return matrix_kappas, antecedent_kappas, value_kappas, G_kappas, S, X, Q, constraints
    else:
        problem.solve(solver=cp.MOSEK)
        return problem.value