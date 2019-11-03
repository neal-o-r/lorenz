import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import ortho_group


def f(state, t):
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


def lorenz():

    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, 40.0, 0.01)

    return odeint(f, state0, t)


def project(A, e1, e2):
    x = A @ e1
    y = A @ e2
    return x, y


def random_bases():
    return ortho_group.rvs(3)[:, :2].T


if __name__ == "__main__":
    e1, e2 = random_bases()

    A = lorenz()

    x, y = project(A, e1, e2)

    plt.plot(x, y, 'k', alpha=0.5, lw=0.5)
    plt.axis('off')
    plt.show()
