import jax.numpy as jnp
import jax
import numpy as np

def getCStar(x, y, pixelWidth, pixelHeight, width, height, rs) -> jnp.array:
    return jnp.array([
        height * (1 - (2 * y) / (pixelHeight - 1)),
        width * (-1 + (2 * x) / (pixelWidth - 1)),
        rs
    ])


def getC(x, y, pixelWidth, pixelHeight, width, height, rs):
    cStar = getCStar(x, y, pixelWidth, pixelHeight, width, height, rs)
    norm = jnp.linalg.vector_norm(cStar, ord=2)
    return cStar / norm

def getCalibrationMatrix(pixelWidth, pixelHeight, viewAngles, rs):
    height = jnp.tan(viewAngles[0] / 2)
    width = jnp.tan(viewAngles[1] / 2)

    XX = np.empty((pixelHeight, pixelWidth), np.float16)
    YY = np.empty((pixelHeight, pixelWidth), np.float16)
    for i in range(pixelHeight):
        for j in range(pixelWidth):
            XX[i, j] = np.float16(j)
            YY[i, j] = np.float16(i)

    vGetC = jax.vmap(getC, in_axes=(0,0,None,None,None,None,None))
    vvGetC = jax.vmap(vGetC, in_axes=(0,0,None,None,None,None,None))

    gradGetC = jax.jacfwd(getC, argnums=[0,1])
    vgradGetC = jax.vmap(gradGetC, in_axes=(0,0,None,None,None,None,None))
    vvgradGetC = jax.vmap(vgradGetC, in_axes=(0,0,None,None,None,None,None))

    calibrationMatrix = vvGetC(XX, YY, pixelWidth, pixelHeight, width, height, rs)
    grad = vvgradGetC(XX, YY, pixelWidth, pixelHeight, width, height, rs)
    
    return calibrationMatrix, grad[0], grad[1]

if __name__ == "__main__":
    pixelWidth = 3
    pixelHeight = 2
    viewAngles = [45, 30]
    rs = 1

    CCM, Cx, Cy = getCalibrationMatrix(pixelWidth, pixelHeight, viewAngles, rs)
