import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go


# Define governing equations
# @profile
def mLorenz(y, t, a, b, c):
    ''' Lorenz attractor modified to have 3 lobes'''
    x, y, z = y
    r = np.sqrt(x**2 + y**2)
    dxdt = 1/3 * (-(a + 1) * x + a - c + y * z)\
        + ((1 - a) * (x**2 - y**2) + 2 * (a + c - z) * x * y) / (3 * r)
    dydt = 1/3 * ((c - a - z) * x - (a + 1) * y)\
        + (2 * (a - 1) * x * y + (a + c - z) * (x**2 - y**2)) / (3 * r)
    dzdt = 1/2 * (3 * x**2 * y - y**3) - b * z
    return [dxdt, dydt, dzdt]


@profile
def make_multiple_Lorenz(N_trials):
    sols = []
    for n in range(N_trials):
        y0_ = [y_ + np.random.randn() for y_ in y0]
        sol = odeint(mLorenz, y0_, t, args=beta)
        sols.append(sol)
    return sols


@profile
def make_plotlies():
    fig = go.Figure([
        go.Scatter3d(name=str(n),
                    x=sols[n][:, 0], y=sols[n][:, 1], z=sols[n][:, 2],
                    line={'width': 2},
                    marker={'size': 1e-5})
        for n in range(len(sols))]
    )
    fig.update_layout(
        width=800,
        height=700,
        autosize=False,
        scene=dict(
            camera=dict(
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                eye=dict(
                    x=0,
                    y=1.0707,
                    z=1,
                )
            ),
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode='manual'
        ),
    )
    return fig


@profile
def set_params():
    beta = (10, 8/3, 137/5)
    y0 = [-8, 4, 10]
    T = 50
    t = np.linspace(0, T, int(T * 200))
    return beta, y0, T, t


# Set parameters, time, & IC
beta, y0, T, t = set_params()

# Integrate the ODE
sols = make_multiple_Lorenz(5)

# Plot the results
fig = make_plotlies()
fig.show()
