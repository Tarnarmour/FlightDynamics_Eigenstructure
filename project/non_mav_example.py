import numpy as np
import control as ct
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from project.eigenstructure_assignment import assign

A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [-5, 2, -1, 1],
              [-3, -3, -1, -1]])

B = np.array([[0, 0],
              [0, 0],
              [1., 0],
              [0, 1.]])

E, V = np.linalg.eig(A)

des_poles = [-3+3j, -3-3j, -2+2j, -2-2j]

K = np.asarray(ct.place(A, B, des_poles))

E1, V1 = np.linalg.eig(A - B @ K)

Vdes = np.array([[1.+1j, 0.0, 1+1j, 0.1],
                 [1-1j, 0.0, 1-1j, 0.1],
                 [0.0, 1+1j, 0.1, 1+1j],
                 [0.0, 1-1j, 0.1, 1-1j]]).T

lam = np.array([-3+3j, -3-3j, -2+2j, -2-2j])

D1 = np.array([[1, 0, 0, 0],
               [0, 0, 1, 0]])

D2 = np.array([[1, 0, 0, 0],
               [0, 0, 1, 0]])

D3 = np.array([[0, 1, 0, 0],
               [0, 0, 0, 1]])

D4 = np.array([[0, 1, 0, 0],
               [0, 0, 0, 1]])


D = [D1, D2, D3, D4]
K_assign, E2, V2 = assign(A, B, des_poles, Vdes, D)

with np.printoptions(precision=4, floatmode='maxprec', suppress=True, linewidth=500):
    print("Open Loop:")
    print(f"Eigenvalues:\n{E}\nEigenvectors:")
    for v in V.T:
        print(v)

    print("\nClosed Loop:")
    print(f"Eigenvalues:\n{E1}\nEigenvectors:")
    for v in V1.T:
        print(v)
    print("K:")
    print(K, "\n")

    print("\nEigenstructure Assignment using Assign:")
    print("Eigenvalues: ", E2)
    print("Desired Eigenvectors:")
    for v in Vdes.T:
        print(v)
    print("Achieved Eigenvectors:")
    for v in V2.T:
        print(v)
    print("K:")
    print(K_assign)


def f1(x, t):
    xd = A @ x + B @ -K @ x
    return xd

def f2(x, t):
    xd = A @ x + B @ -K_assign @ x
    return xd


if __name__ == '__main__':

    x0 = np.array([5, 10, 0, 0])
    ts = np.linspace(0, 10, 1000)
    y1 = odeint(f1, x0, ts)
    y2 = odeint(f2, x0, ts)

    fig, ax = plt.subplots()

    ax.plot(ts, y1[:,0], color=np.array([0, 0, 1.0, 1]))
    ax.plot(ts, y1[:,1], color=np.array([0, 0, 0.5, 1]))
    ax.plot(ts, y2[:, 0], color=np.array([1.0, 0, 0, 1]))
    ax.plot(ts, y2[:, 1], color=np.array([0.5, 0, 0, 1]))

    plt.legend(['x1, PP', 'x2, PP', 'x1, EA', 'x2, EA'])

    plt.pause(0.1)
    plt.show()
