import numpy as np


def monomial_vector(d, n):
    """
    Generate monomial vectors for n variables up to degree less than or equal to d (defined by recursion).
    :param d: maximum degree of the monomial
    :param n: number of variable
    :return: List of monomials in the form [x1^a1 * x2^a2 * ... * xn^an] where a1+a2+...+an<=d.
    """

    if d == 0:
        return [[0] * n]
    if n == 1:
        return [[i] for i in range(d + 1)]
    result = []
    for i in range(d + 1):
        submonomials = monomial_vector(d - i, n - 1)
        for submonomial in submonomials:
            if sum(submonomial) <= d - i:
                result.append([i] + submonomial)
    return result # it remains a list because of the recursion

def kappas(monomials):
    """
    This function returns the matrix kappas constructed from the monomials. It also builds the possible antecedents
    for each kappas and the list for each kappas.
    :param monomials: np.ndarrray (and not a list)
    :return:
    """
    d = np.max(monomials)
    n = monomials.shape[1]
    d_max = n * d
    size_monom = monomials.shape[0]
    matrix_kappas = np.zeros((size_monom, size_monom, n))
    antecedent_kappas = {}
    value_kappas = []
    for i in range(size_monom):
        for j in range(size_monom):
            for k in range(n):
                matrix_kappas[i][j][k] = monomials[i][k] + monomials[j][k]
            name = f'({monomials[i] + monomials[j]})'
            if name in antecedent_kappas.keys():
                antecedent_kappas[name].append([i, j])
            else:
                antecedent_kappas[name] = [[i, j]]
                value_kappas.append(monomials[i] + monomials[j])
    assert len(antecedent_kappas) == len(value_kappas)
    return matrix_kappas, antecedent_kappas, np.array(value_kappas)

def decomposition_SDP_to_monomials(Q, antecedent_kappas):
    """
    Given a multivariate polynomial Q and the dictionary of the antecedents of kappas, this function return
    Q_kappas (in the right basis).
    :param Q: np.ndarray
    :param antecedent_kappas: dict
    :return:
    """
    Q_kappas = []
    u = 0
    for k in antecedent_kappas.keys():
        idx = antecedent_kappas[k]
        Q_kappas.append(sum(Q[id[0], id[1]] for id in idx))
        u += 1
    return Q_kappas

def new_kappas(G, matrix_kappas, value_kappas):
    """
    Given a family of polynomials and the matrix of kappas defining the current basis, this function
    returns the new values for kappas that are not in the current basis.
    :param G: np.ndarray
    :param matrix_kappas: np.ndarray
    :param value_kappas: np.ndarray
    :return:
    """
    # count the non zero arguments of G
    nonzeros = []
    degree = np.max(matrix_kappas) / 2
    for i in range(len(G)):
        nz_arg = np.array(np.nonzero(G[i])).T
        for j in range(len(nz_arg)):
            new_prod = matrix_kappas[nz_arg[j][0], nz_arg[j][1]]
            nonzeros.append(new_prod)
    nonzeros = np.array(nonzeros)
    # build the dictionnary of new kappas
    new_kappas = {}
    for i in range(len(value_kappas)):
        for j in range(len(nonzeros)):
            new_k = value_kappas[i] + nonzeros[j]
            if np.sum(new_k) > 2 * degree:
                name = f'({new_k})'
                if name not in new_kappas.keys():
                    new_kappas[name] = new_k
    return new_kappas