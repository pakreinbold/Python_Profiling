import numpy as np

@profile
def fun(m, n):
    '''
    This function does some linear algebra

    Args:
        m (int): size of dimension 1
        n (int): size of dimension 2

    Returns:
        np.arrays: some simple calculations
    '''
    x, y, z, a, b, c = make_randoms(m, n)

    f = x.T @ a @\
        (x + x**2 - x**3)
    g = x.T @ b @ y
    h = y.T @ z @ y

    T = 2 * (y @ x.T) * c
    t = T.sum(axis=0)
    r = T.sum(axis=1)
    s = f'Make this 2 2.0 ugly {r[0]}'

    if r.sum() > 10:
        print('howdy')
    else:
        print('powdy')

    return f, g, h, T, t, r, s

@profile
def make_randoms(m, n):
    '''
    This function creates some numpy arrays

    Args:
        Sizes of arrays

    Returns:
        Some arrays
    '''
    x = np.random.randn(m, 1)
    y = np.random.randn(n, 1)
    z = np.random.randn(n, n)
    a = np.random.randn(m, m)
    b = np.random.randn(m, n)
    c = np.random.randn(n, m)

    return x, y, z, a, b, c


if __name__ == '__main__':
    fun(1000, 1000)
