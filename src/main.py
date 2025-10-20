# Header
# Using scalene for profiling 
# scalene --html --outfile file.html main.py

# Imports
import argparse
import jax
import jax.numpy as jnp
import time
from fileOperations import writeToFile, createFolderAndUpdateGitIgnore, loadSettings, loadCalibration
import os
import cv2





import logging

# Configure the logging module
logging.basicConfig(
    level=logging.DEBUG,  # Log all messages (DEBUG and above)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log"),  # Log to a file named "log"
        logging.StreamHandler()      # Also log to console (optional)
    ]
)

# Example usage
logger = logging.getLogger(__name__)

# Body

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allowed options")

    parser.add_argument(
        "-h", "--help", action="help", help="Produce help message"
    )
    parser.add_argument(
        "-f", "--startTime", type=float, default=0,
        help="Where to start with event consideration"
    )
    parser.add_argument(
        "-e", "--endTime", type=float, default=10,
        help="Where to end with event consideration"
    )
    parser.add_argument(
        "-t", "--timeStep", type=float, default=0.0460299576597383,
        help="Size of the event frames"
    )
    parser.add_argument(
        "-s", "--resourceDirectory", type=str, default="boxes_rotation",
        help="Which dataset to use, searches in res directory"
    )
    parser.add_argument(
        "-r", "--resultsDirectory", type=str, default="boxes_rotation",
        help="Where to store the results, located in output directory"
    )
    parser.add_argument(
        "-a", "--addTime", type=bool, default=False,
        help="Add time to output folder?"
    )
    parser.add_argument(
        "-i", "--startIndex", type=int, default=0,
        help="With what index to start for the images"
    )
    parser.add_argument(
        "-fr", "--fuseR", type=bool, default=False,
        help="Fuse with imu.txt?"
    )
    parser.add_argument(
        "-fi", "--fuseI", type=bool, default=False,
        help="Fuse with images?"
    )

    args = parser.parse_args()

    start = time.perf_counter()

    resultsPath = createFolderAndUpdateGitIgnore(args.resultsDirectory)
    logger.info(f"Create folder {resultsPath}")

    calibrationPath = os.path.join("res", args.resourceDirectory, "calib.txt")
    eventPath = os.path.join("res", args.resourceDirectory, "events.txt")
    imuPath = os.path.join("res", args.resourceDirectory, "imu.txt")
    imagesPath = os.path.join("res", args.resourceDirectory, "images.txt")
    settingsPath = os.path.join("res", args.resourceDirectory, "settings.txt")

    rotVelocityPath = os.path.join(resultsPath, "R.txt")
    VLossPath = os.path.join(resultsPath, "VLoss.txt")

    settings = loadSettings(os.path.join("res", "settings.json"))
    
    vectorPermutation = [0,1,2]
    randomKey = jax.random.PRNGKey(99)
    randomKey, subKey1, subKey2 = jax.random.split(randomKey, num=3)

    # Tensor definitions
    opticalFlow = jnp.zeros((settings["height"], settings["width"], 2), jnp.float16) 
    opticalFlowVis = jnp.zeros((settings["height"], settings["width"]), jnp.float16)
    spatialGradient = jnp.zeros((settings["height"], settings["width"], 2), jnp.float16) 
    DiffSpatialGradient = jnp.zeros((settings["height"], settings["width"]), jnp.float16)
    gradientDiffSpatialGradient = jnp.zeros((settings["height"], settings["width"]), jnp.float16)
    deltaIntensity = jnp.zeros((settings["height"], settings["width"], 2), jnp.float16) 
    intensity = jnp.ones((settings["height"], settings["width"]), jnp.float16) * 128.0
    decayTimeSurface = jnp.ones((settings["height"], settings["width"]), jnp.float16) * settings["startTime"]
    rotationalVelocity = jax.random.uniform(subKey1, shape=(3,1), minval=-10.0, maxval=10.0)
    
    calibrationData = loadCalibration(os.path.join("res", "calib.json"), height=settings.height, width=settings.width)

    

















    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds")