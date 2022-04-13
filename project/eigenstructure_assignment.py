import numpy as np
from numpy.linalg import eig, inv, lstsq, norm, solve


def assign(A, B, des_poles, Vd, D):

    n = B.shape[0]
    m = B.shape[1]

    I = np.eye(n)

    Va = np.zeros((n, n), dtype=Vd.dtype)
    U = np.zeros((m, n), dtype=Vd.dtype)

    for i in range(n):

        e = des_poles[i]
        vd = Vd[:, i]
        d = D[i]
        k = d.shape[0]

        M = np.block([[e*I - A, B],
                      [d, np.zeros((k, m))]])

        N = np.hstack([np.zeros((n,)), d @ vd])

        if m == k:
            c = solve(M, N)
        else:
            c, res, rank, sing = lstsq(M, N, rcond=None)
        va = c[0:n]
        u = c[n:None]

        Va[:, i] = va
        U[:, i] = u

    K = U @ inv(Va)

    if norm(K.imag) > 1e-8:
        print("It's all in your head")
        return None
    else:
        K = K.real

    E, V = eig(A - B @ K)

    return K, E, V
