import numpy as np
from jax import numpy as jnp
from typing import List

class Event:
    """
    A base class to represent a generic event.
    """
    def __init__(self, time: int = 0):
        """
        Initialize the event.

        Args:
            time (float): The timestamp of the event.
        """
        self.time = time

    def __repr__(self):
        """
        Return a string representation of the event.
        """
        return f"{self.__class__.__name__}(time={self.time})"


class NeuromorphicCameraEvent(Event):
    """
    A class to represent a neuromorphic camera event.
    """
    def __init__(self,
                 time: int,
                 x: int,
                 y: int,
                 polarity: int):
        """
        Initialize the neuromorphic camera event.

        Args:
            time (float): The timestamp of the event.
            x (int): The x-coordinate of the pixel.
            y (int): The y-coordinate of the pixel.
            polarity (int): The polarity of the event (1 for ON, -1 for OFF).
        """
        super().__init__(time)
        self.x = x
        self.y = y
        self.polarity = polarity

    def __repr__(self):
        """
        Return a string representation of the neuromorphic camera event.
        """
        return (f"{self.__class__.__name__}(time={self.time}, x={self.x}, "
                f"y={self.y}, polarity={self.polarity})")


class NormalCameraEvent(Event):
    """
    A class to represent a normal camera event where the entire image changes.
    """
    def __init__(self,
                 time: int, 
                 image: jnp.array):
        """
        Initialize the normal camera event.

        Args:
            time (float): The timestamp of the event.
            image (np.ndarray or jnp.ndarray): The image data.
        """
        super().__init__(time)
        self.image = image

    def __repr__(self):
        """
        Return a string representation of the normal camera event.
        """
        return f"{self.__class__.__name__}(time={self.time}, image_shape={self.image.shape})"


class IMUEvent(Event):
    """
    A class to represent an IMU event.
    """
    def __init__(self,
                 time:int, 
                 velocity,
                 acceleration):
        """
        Initialize the IMU event.

        Args:
            time (float): The timestamp of the event.
            velocity (tuple): The velocity as a 3D vector (vx, vy, vz).
            acceleration (tuple): The acceleration as a 3D vector (ax, ay, az).
        """
        super().__init__(time)
        self.velocity = velocity
        self.acceleration = acceleration

    def __repr__(self):
        """
        Return a string representation of the IMU event.
        """
        return (f"{self.__class__.__name__}(time={self.time}, velocity={self.velocity}, "
                f"acceleration={self.acceleration})")
    
class CalibrationData:
    """
    Images of a camera can be distorted by said camera. If certain parameters about the camera are known, the images can 
    be undistorted.
    """
    def __init__(self,
                 focalPoint: np.array,
                 cameraMatrix: np.array,
                 distortionCoefficients: np.array,
                 viewAngles: np.array):
        self.focalPoint = focalPoint
        self.cameraMatrix = cameraMatrix
        self.distortionCoeffiecents = distortionCoefficients
        self.viewAngles = viewAngles

    def __repr__(self):
        """
        Return a string representation of the calibration data.
        """
        return (f"{self.__class__.__name__}(focalPoint={self.focalPoint}, cameraMatrix={self.cameraMatrix}, distortionCoefficients={self.distortionCoeffiecents}, viewAngles={self.viewAngles}")