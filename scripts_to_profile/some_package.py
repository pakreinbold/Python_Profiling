import numpy as np


@profile
def do_some_math(m, n):
    x = np.random.randn(1, m)
    A = np.random.randn(m, n)
    y = np.random.randn(n, 1)

    return x @ A @ y
