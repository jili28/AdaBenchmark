import jax.numpy as np


def hokalman(G, T, n, t1, t2):
    """
    Implements Ho-
    assumes G = [D, G0, G1, ... G_{T-2}]

    :return:
    """
    m = G.shape[0]
    p = G.shape[1] // T  # need to find something else
    # construct H s.t. (i, j)-block corresponds G_{i+j-2}
    # if t1 + t2 <= t --> guarantee full rank
    l = [[G[:, (1 + i) * p:(t2 + i + 1) * p], np.zeros((m, p))] for i in range(t1)]
    # l = [ [G[:, i*p:t2+)*p], np.zeros((m, (1+i)*p))] for i in range(t1)]  #alternative if we want to fully include all columns

    h = np.block(l)

    h_m = h[:, :p * t2]
    u, s, vh = np.linalg.svd(h_m)
    o = u[:, :n] * np.sqrt(s[:n])
    q = np.dot(np.diag(np.sqrt(s[:n])), vh[:n, :])

    A = np.matmul(np.matmul(np.linalg.pinv(o), h[:, p:]), np.linalg.pinv(q))  # can probably to closed form solution
    return A


def systemident(N, p, K=None):
    """
    Alternative implementation of Ho-Kalman procedure
    :param N:
    :param p:
    :param K:
    :return:
    """
    k = int(N.shape[1] / p)
    # print(k)
    # print(p)
    C0 = N[:, :p * (k - 1)]
    C1 = N[:, p:]
    A = np.matmul(C1, np.linalg.pinv(C0))
    B = C0[:, :p]

    if K is not None:
        A = A + np.matmul(B, k)
    return A, B #B, in expectation correct, but else not really


def check(A, B, t, t1, t2):
    l = [np.dot(np.linalg.matrix_power(A, i), B) for i in range(t)]
    l = [A] + l
    G = np.block(l)


# from ho_kalman import hokalman
# import numpy as np
#
# A = np.array([[2, 0], [0, 1]])
# B = np.array([[2, 1], [2, 3]])
# t = 10
# t1, t2 = 4, 4
# l = [A] + [np.dot(np.linalg.matrix_power(A, i), B) for i in range(t - 1)]
# G = np.block(l)
# hokalman(G, 10, 2, 4, 5)

# there is some necessity for weighting
