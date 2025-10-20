import jax 
import jax.numpy as jnp

def method(x, y, z):
    return x*y - z

if __name__ == "__main__":
    X = jnp.array([
        [1,2,3],
        [4,5,6]
    ])
    Y = jnp.array([
        [11,22,33],
        [43,54,65]
    ])
    z = .5

    vMethod = jax.vmap(method, in_axes=(0,0,None))
    vvMethod = jax.vmap(vMethod, in_axes=(0,0,None))
    print(vvMethod(X, Y, z))

