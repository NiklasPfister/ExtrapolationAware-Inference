import jax
import jax.numpy as jnp
import numpy as np


# First order directional derivative
def Dv(f, v):
    def fv(vx, x): return f((x + vx * v).reshape(1, -1)).reshape(())
    tmp = jax.grad(fv)
    def Dfv(x): return(tmp(0.0, x.reshape(-1, 1)))
    return jax.vmap(Dfv, 0)


# Second orders directional derivative
def DDv(f, v):
    def Dfv(vx, x): return Dv(f, v)((x + vx * v).reshape(1, -1)).reshape(())
    tmp = jax.grad(Dfv)
    def DDfv(x): return(tmp(0.0, x.reshape(-1, 1)))
    return jax.vmap(DDfv, 0)


##
# List of example functions - one-dimensional (also work multi-dimensional)
##

def linear(x):
    def inner(x): return x[0]*2
    return jax.vmap(inner, 0)(x)


def sin(x):
    def inner(x): return jnp.sin(jnp.pi*x[0])
    return jax.vmap(inner, 0)(x)


def slow_sin(x):
    def inner(x): return jnp.sin(jnp.pi*x[0]/4)
    return jax.vmap(inner, 0)(x)


def expit(x):
    def inner(x): return 2/(1+jnp.exp(-(2*(x[0]-0.5))))
    return jax.vmap(inner, 0)(x)


def quadratic(x):
    def inner(x): return x[0]**2
    return jax.vmap(inner, 0)(x)


##
# List of example functions - multi-dimensional
##

def x0x1(x):
    def inner(x): return x[0] * x[1]
    return jax.vmap(inner, 0)(x)


def sqrt_x0x1(x):
    def inner(x): return jnp.sqrt(jnp.abs(x[0] * x[1]))*jnp.sign(x[0] * x[1])
    return jax.vmap(inner, 0)(x)


def rotated_linear(x):
    def inner(x): return x[0] + x[1]
    return jax.vmap(inner, 0)(x)


def rotated_quadratic(x):
    def inner(x): return (x[0] + x[1])**2
    return jax.vmap(inner, 0)(x)


def rotated_expit(x):
    def inner(x): return 2/(1+jnp.exp(-(2*(x[0]+x[1]-0.5))))
    return jax.vmap(inner, 0)(x)


def linear_fun_generator(ab):
    return lambda x: (x - ab[2])*ab[1]+ab[0]


def piecewise_linear(x, dvec, grid):
    xx = x[:, 0]
    cl = []
    fl = []
    ab = [0, 0, 0]
    for i in range(len(grid)-1):
        # Compute and append a "condition" interval
        cl.append(jnp.logical_and(xx >= grid[i], xx <= grid[i+1]))
        # Update parameters
        ab[1] = dvec[i]
        ab[2] = grid[i]
        # Create a linear function for the interval
        fl.append(linear_fun_generator(ab * 1))
        # Update offset
        ab[0] += (grid[i+1]-grid[i])*ab[1]
    return(jnp.piecewise(xx, condlist=cl, funclist=fl))


def rotated_piecewise_linear(x, dvec, grid):
    xx = x[:, 0]/2 + x[:, 1]/2
    cl = []
    fl = []
    ab = [0, 0, 0]
    for i in range(len(grid)-1):
        # Compute and append a "condition" interval
        cl.append(jnp.logical_and(xx >= grid[i], xx <= grid[i+1]))
        # Update parameters
        ab[1] = dvec[i]
        ab[2] = grid[i]
        # Create a linear function for the interval
        fl.append(linear_fun_generator(ab * 1))
        # Update offset
        ab[0] += (grid[i+1]-grid[i])*ab[1]
    return(jnp.piecewise(xx, condlist=cl, funclist=fl))


# Function to sample multi-dimensional X-distribution
def sample_X(n, bds):
    Xtrain = np.empty((n, len(bds)))
    Xtest = np.empty((n, len(bds)))
    for j, bd in enumerate(bds):
        if bd == 0:
            Xtrain[:, j] = np.random.uniform(-1, 2, n)
            Xtest[:, j] = np.random.uniform(-2, -1, n)
        elif bd == 1:
            mix = np.random.binomial(1, 1/3, n)
            Xtrain[:, j] = (mix * np.random.uniform(-2, -1, n) + (1-mix) *
                            np.random.uniform(0, 2, n))
            Xtest[:, j] = np.random.uniform(-1, 0, n)
        elif bd == 2:
            mix = np.random.binomial(1, 1/3, n)
            Xtrain[:, j] = ((1-mix) * np.random.uniform(-2, 0, n) + mix *
                            np.random.uniform(1, 2, n))
            Xtest[:, j] = np.random.uniform(0, 1, n)
        elif bd == 3:
            Xtrain[:, j] = np.random.uniform(-2, 1, n)
            Xtest[:, j] = np.random.uniform(1, 2, n)
    return Xtrain, Xtest
