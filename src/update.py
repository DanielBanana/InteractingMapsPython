import jax.numpy as jnp
from jax import Array
from numpy.typing import NDArray
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

    cameraCalibrationMatrix = vvGetC(XX, YY, pixelWidth, pixelHeight, width, height, rs)
    grad = vvgradGetC(XX, YY, pixelWidth, pixelHeight, width, height, rs)
    
    return cameraCalibrationMatrix, grad[0], grad[1]

def setupRUpdate(cameraCalibrationMatrix: Array):
    rows, cols, depth = cameraCalibrationMatrix.shape
    identity = jnp.identity(3, jnp.float16)
    points = np.zeros((rows, cols, 3))
    vecB = np.zeros((3))

    vFunc = jax.vmap(jax.vmap(lambda c, i: i - jnp.outer(c, c), in_axes=(0, None)), in_axes=(0, None)) # map over the two first dimensions of the first input (C).
    vI = vFunc(cameraCalibrationMatrix, identity)
    matA = jnp.sum(vI, axis=[0,1])
    return matA, vecB, vI, points

def updateFG(F: NDArray, V: float, G: NDArray, lr: float, weightFG: float, gamma: float=255):

    normG = jnp.linalg.norm(G, ord=2)
    if (not np.isclose(normG, 0.0, atol=1e-6)):
        newF = F - G/normG**2 * (V/normG + np.dot(F, G))
        newF = (1- weightFG) * F + lr * weightFG * newF
        F = np.clip(newF, -gamma, gamma)
    return F

def updateGI(G, gradI, lr, weightGI, gamma):
    G = np.clip((1-weightGI) * G + lr * weightGI * gradI, -gamma, gamma)
    return G

def updateGradientLocal3D(arr, grad, i, j):
    """
    Update the gradient after modifying arr[i, j] to new_value.

    Args:
        arr: The 3D array.
        grad_y: Current gradient in the y-direction (rows).
        grad_x: Current gradient in the x-direction (columns).
        i: Row index of the modified element.
        j: Column index of the modified element.
        new_value: New value for arr[i, j].

    Returns:
        Updated grad_y and grad_x.
    """
    # Define the neighborhood (3x3 window)
    i_min, i_max = max(0, i - 1), min(arr.shape[0], i + 2)
    j_min, j_max = max(0, j - 1), min(arr.shape[1], j + 2)

    # Extract the neighborhood
    neighborhood = arr[i_min:i_max, j_min:j_max]

    # Recompute the gradient for the neighborhood
    local_grad_y, local_grad_x = np.gradient(neighborhood, axis=(0,1))

    # Update the global gradient arrays
    grad[i_min:i_max, j_min:j_max,0] = - local_grad_y[:,:,0]
    grad[i_min:i_max, j_min:j_max,1] = local_grad_x[:,:,1]

    return grad

def updateGradientLocal2D(arr, grad, i, j):
    """
    Update the gradient after modifying arr[i, j] to new_value.

    Args:
        arr: The 3D array.
        grad_y: Current gradient in the y-direction (rows).
        grad_x: Current gradient in the x-direction (columns).
        i: Row index of the modified element.
        j: Column index of the modified element.
        new_value: New value for arr[i, j].

    Returns:
        Updated grad_y and grad_x.
    """
    # Define the neighborhood (3x3 window)
    i_min, i_max = max(0, i - 1), min(arr.shape[0], i + 2)
    j_min, j_max = max(0, j - 1), min(arr.shape[1], j + 2)

    # Extract the neighborhood
    neighborhood = arr[i_min:i_max, j_min:j_max]

    # Recompute the gradient for the neighborhood
    local_grad_y, local_grad_x = np.gradient(neighborhood, axis=(0,1))

    # Update the global gradient arrays
    grad[i_min:i_max, j_min:j_max,0] = -local_grad_y # invert the y gradient because the gradient gets caculated from top to bottom, but we plot from bottom to top (regarding y values)
    grad[i_min:i_max, j_min:j_max,1] = local_grad_x  # x values always go from right to left

    return grad

def updateGIDifferenceGradient(G, gradI, GIDifference, GIDifferenceGradient, y, x):
    GIDifference[y,x] = G[y,x] - gradI[y,x]
    # Update the gradient in a 3x3 window arround the value at y, x
    return updateGradientLocal3D(GIDifference, GIDifferenceGradient, y, x)


def updateIV(I, V: float, minValue, maxValue, lr, weightIV, decayTimeSurface, time, neutralPotential, decayParameter):
    I = contribute(I, V, minValue, maxValue, lr, weightIV)
    I = exponentialDecay(I, decayTimeSurface, time, neutralPotential, decayParameter)
    return I

def updateIG(I, GIDifferenceGradient, lr, weightIG):
    return (1 - weightIG) * I + lr * weightIG * (-GIDifferenceGradient[0] - GIDifferenceGradient[1])

def contribute(I, V, minValue, maxValue, lr, weightIV):
    I = np.clip(I + weightIV * lr * V, minValue, maxValue)
    return I

def linearDecay(I, decayTimeSurface, time, neutralPotential, decayParameter):
    valueDifference = I - neutralPotential
    timeDifference = time - decayTimeSurface
    I = I - np.min(valueDifference * timeDifference * decayParameter, valueDifference)
    return I

def exponentialDecay(I, decayTimeSurface, time, neutralPotential, decayParameter):
    valueDifference = I - neutralPotential
    timeDifference = time - decayTimeSurface
    if valueDifference.ndim > timeDifference.ndim:
        timeDifference = timeDifference[..., np.newaxis]
    I = neutralPotential + np.clip(valueDifference * np.exp(-timeDifference * decayParameter), -np.abs(valueDifference), np.abs(valueDifference))
    return I
    
def updateFR(F, CCM, Cx, Cy, R, lr, weightFR, gamma):
    cross = np.cross(R, CCM)
    update = m32(cross, Cx, Cy)
    return np.clip((1 - weightFR) * F + weightFR * lr * update, -gamma, gamma)

def updateRF(R, F, CCM, Cx, Cy, A, b, Iminus, oldPoint, lr, weightRF):
    newF = m23P(F, Cy, Cx)
    point = np.cross(CCM, newF)
    b = b - Iminus @ oldPoint + Iminus @ point
    oldPoint = point.copy()

    solution = np.linalg.solve(A, b)
    return (1 - weightRF) * R + weightRF * lr * solution, b, oldPoint

def updateRImu(R, rotVelImu, lr, weightRImu):
    return (1 - weightRImu) * R + weightRImu * lr * rotVelImu


def m23A(F, Cx, Cy):
    return jnp.einsum("ij,ijk->ijk", F[:,:,1], Cx[:,:]) + jnp.einsum("ij,ijk->ijk", F[:,:,0], Cy[:,:])

def m23P(F, Cx, Cy):
    return F[1] * Cx + F[0] * Cy


def m32(vec3, Cx, Cy):
    out = np.empty((*vec3.shape[0:2],2))

    C1 = jnp.cross(Cx, Cy)
    C2 = jnp.cross(Cy, C1)
    dotTemp = np.einsum("ijk,ijk->ij", vec3, C2)
    signTemp = jnp.sign(dotTemp)
    d1 = vectorDistance(vec3, Cy)
    d2 = vectorDistance(Cx, Cy)
    out[:,:,1] = signTemp * d1/d2

    C1 = jnp.cross(Cy, Cx)
    C2 = jnp.cross(Cx, C1)
    dotTemp = np.einsum("ijk,ijk->ij", vec3, C2)
    signTemp = jnp.sign(dotTemp)
    d1 = vectorDistance(vec3, Cx)
    d2 = vectorDistance(Cy, Cx)
    out[:,:,0]= signTemp * d1/d2
    return out

def vectorDistance(vec1, vec2):
    cross = jnp.cross(vec1, vec2)
    norm1 = jnp.linalg.norm(cross, axis=-1)
    norm2 = jnp.linalg.norm(vec2, axis=-1)
    return norm1/norm2

def intensityStep(temporalGradient):
    pass

if __name__ == "__main__":
    pixelWidth = 8
    pixelHeight = 8
    viewAngles = [45, 30]
    rs = 1

    CCM, Cx, Cy = getCalibrationMatrix(pixelWidth, pixelHeight, viewAngles, rs)

    A, b, Iminus, points = setupRUpdate(CCM)

    F = np.random.rand(pixelHeight,pixelWidth,2)
    G = np.random.rand(pixelHeight,pixelWidth,2)
    I = np.random.rand(pixelHeight,pixelWidth,1)*4 + 128
    V = 4

    updateFG(F, V, G, 0, 0, 1, 0.3, 255)

    gradI = np.stack(np.gradient(I, axis=(0,1)), axis=2).squeeze()
    GIDifference = G-gradI
    GIDifferenceGradient_y, GIDifferenceGradient_x = np.gradient(GIDifference, axis=(0,1))
    # from the gradient_x we only need the gradient in x direction and from the gradient_y the gradient in y direction
    GIDifferenceGradient = np.stack((GIDifferenceGradient_y[:,:,1], GIDifferenceGradient_x[:,:,0]), 2)

    G[2,2] = 1
    gradI[2,2] = 0

    updateGIDifferenceGradient(G, gradI, GIDifference, GIDifferenceGradient, y=2, x=2)

    print(GIDifference)
    print(GIDifferenceGradient)

    decayTimeSurface = np.random.rand(8,8)
    decaParameter = 1e2
    y = 2
    x = 2
    time = decayTimeSurface[y, x] + 1e-3
    updateIV(I, V, y=y, x=x, minValue=0, maxValue=255, lr=1, weightIV=0.5, decayTimeSurface=decayTimeSurface, time=time, decayParameter=decaParameter, neutralPotential=128)
    print(I[2,2])

    R = np.random.rand(3)
    lr = 1.0
    weightFR = 0.8
    gamma = 255
    updateFR(F, CCM, Cx, Cy, R, lr, weightFR, gamma)

    m23A(F, Cx, Cy)

    updateRF(R, F, CCM, Cx, Cy, A, b, Iminus, points, lr, weightRF = 0.8, y=y, x=x)

    print(R)
    R = updateRImu(R, np.random.rand(3), lr, weightRImu=0.8)
    print(R)