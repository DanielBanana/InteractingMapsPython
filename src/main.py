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
import numpy as np
from fileOperations import readEvents
from update import exponentialDecay, setupRUpdate, getCalibrationMatrix, updateGradientLocal3D, updateGradientLocal2D, updateIG, updateFG, updateFR, updateGI, updateGIDifferenceGradient, updateIV, updateRF, updateRImu
import polars as pl
from imaging import *



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

    # parser.add_argument(
    #     "-h", "--help", action="help", help="Produce help message"
    # )
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
        "-s", "--resourceDirectory", type=str, default="shapes_rotation",
        help="Which dataset to use, searches in res directory"
    )
    parser.add_argument(
        "-r", "--resultsDirectory", type=str, default="shapes_rotation",
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

    calibrationPath = os.path.join("res", args.resourceDirectory, "calib.json")
    eventPath = os.path.join("res", args.resourceDirectory, "events.txt")
    imuPath = os.path.join("res", args.resourceDirectory, "imu.txt")
    imagesPath = os.path.join("res", args.resourceDirectory, "images.txt")
    settingsPath = os.path.join("res", args.resourceDirectory, "settings.txt")

    rotVelocityPath = os.path.join(resultsPath, "R.txt")
    VLossPath = os.path.join(resultsPath, "VLoss.txt")

    settings = loadSettings(os.path.join("res", "settings.json"))
    height = settings["height"]
    width = settings["width"]
    gamma = settings["gamma"] # -gamma and gamma are the min and max values for entries in F or G since it makes it easier to plot


    permutation = [0,1,2,3,4,5]
    # randomKey = jax.random.PRNGKey(99)
    # randomKey, subKey1, subKey2 = jax.random.split(randomKey, num=3)

    # Tensor definitions
    opticalFlow = np.random.uniform(-0.1,0.1,(settings["height"], settings["width"], 2)) 
    opticalFlow = np.zeros((settings["height"], settings["width"], 2), np.float16)
    opticalFlow[:,:,1] = 1
    spatialGradient = np.ones((settings["height"], settings["width"], 2)) 
    temporalGradient = np.zeros((settings["height"], settings["width"]), np.float16) 
    DiffSpatialGradient = np.zeros((settings["height"], settings["width"], 2), np.float16)
    gradientDiffSpatialGradient = np.zeros((settings["height"], settings["width"], 2), np.float16)
    deltaIntensity = np.zeros((settings["height"], settings["width"], 2), np.float16) 
    intensity = np.full((settings["height"], settings["width"]), settings["neutralPotential"], np.float16)
    memoryIntensity = np.full((height, width), settings["neutralPotential"], np.float16)

    decayTimeSurface = np.ones((settings["height"], settings["width"]), np.float16) * settings["startTime"]
    # rotationalVelocity = jax.random.uniform(subKey1, shape=(3), minval=-0, maxval=1.0)

    rotationalVelocity = np.random.uniform(-.01,.01,3)
    
    calibrationData = loadCalibration(calibrationPath, height=settings["height"], width=settings["width"])

    CCM, Cx, Cy = getCalibrationMatrix(width, height, calibrationData.viewAngles, 1.0)

    A, b, Iminus, points = setupRUpdate(CCM)

    angularVelocity = np.zeros(3)
    acceleration = np.zeros(3)
    
    lazyEvents = readEvents(settings["eventsPath"], settings["startTime"], settings["endTime"], height=[0,settings["height"]], width=[0,settings["width"]])
    nEvents = lazyEvents.select(pl.len()).collect().item()

    eventIndexOffset = 0
    eventIndexLength = 100
    eventIndexCounter = 0
    updateCounterFR = 0
    updateIterationsFR = 5

    setupRUpdate

    nCamEvent = True # neuromorphic Camera Event; at the moment always true

    batchSize = 1

    while eventIndexLength*eventIndexCounter < nEvents:
        events = lazyEvents.slice(eventIndexLength*eventIndexCounter, length=eventIndexLength).collect()
        print(f"Time: {events.head(1)["time"].item()}")
        
        for event in events.iter_rows():
            updateCounterFR += 1
            if updateCounterFR == eventIndexLength-1:
                updateCounterFR = 0
                permutation += [6]
            
            # np.random.shuffle(permutation)

            if nCamEvent:
                V = event[3] * 2 - 1 # Convert polarity from 0,1 to -1,1
                y = event[2]
                x = event[1]

                temporalGradient[y,x] = V

            for i in range(2):
                for i, step in enumerate(permutation):
                    match step:
                        case 0:
                            intensity[y,x] = updateIV(
                                I=intensity[y,x],
                                V=temporalGradient[y,x],
                                minValue=settings["minPotential"],
                                maxValue=settings["maxPotential"],
                                weightIV=settings["weightIV"],
                                decayTimeSurface=decayTimeSurface[y,x],
                                neutralPotential=settings["neutralPotential"],
                                decayParameter=settings["decayParameter"],
                                lr=settings["lr"],
                                time=event[0]
                            )
                            decayTimeSurface[y,x] = event[0]
                            deltaIntensity = updateGradientLocal2D(intensity, deltaIntensity, y, x)
                            
                        case 1:
                            spatialGradient[y,x] = updateGI(
                                G=spatialGradient[y,x],
                                gradI=deltaIntensity[y,x],
                                lr=settings["lr"],
                                weightGI=settings["weightGI"],
                                gamma=gamma
                            )
                            gradientDiffSpatialGradient= updateGIDifferenceGradient(
                                G=spatialGradient,
                                gradI=deltaIntensity,
                                GIDifference=DiffSpatialGradient,
                                GIDifferenceGradient=gradientDiffSpatialGradient,
                                y=y,
                                x=x
                            )
                        case 2:
                            intensity[y,x] = updateIG(
                                I=intensity[y,x],
                                GIDifferenceGradient=gradientDiffSpatialGradient[y,x],
                                lr=settings["lr"],
                                weightIG=settings["weightIG"]
                            )
                        case 3:
                            opticalFlow[y,x] = updateFG(
                                F=opticalFlow[y,x],
                                V=temporalGradient[y,x],
                                G=spatialGradient[y,x],
                                lr=settings["lr"],
                                weightFG=settings["weightFG"],
                                gamma=gamma
                            )
                        case 4:
                            spatialGradient[y,x] = updateFG(
                                F=spatialGradient[y,x],
                                V=temporalGradient[y,x],
                                G=opticalFlow[y,x],
                                lr=settings["lr"],
                                weightFG=settings["weightGF"],
                                gamma=gamma
                            )
                        case 5:
                            rotationalVelocity, b, points[y,x] = updateRF(
                                R=rotationalVelocity,
                                F=opticalFlow[y,x],
                                CCM=CCM[y,x],
                                Cx=Cx[y,x],
                                Cy=Cy[y,x],
                                A=A,
                                b=b,
                                Iminus=Iminus[y,x],
                                oldPoint=points[y,x],
                                lr=settings["lr"],
                                weightRF=settings["weightRF"],
                            )
                        case 6:
                            for _ in range(updateIterationsFR):
                                opticalFlow = updateFR(
                                    F=opticalFlow,
                                    CCM=CCM,
                                    Cx=Cx,
                                    Cy=Cy,
                                    R=rotationalVelocity,
                                    lr=settings["lr"],
                                    weightFR=settings["weightFR"],
                                    gamma=gamma)
                            permutation.pop(i)
        print(rotationalVelocity)
        vigfPath = os.path.join(resultsPath, f"vigf_{eventIndexCounter}.png")
        plot_VvsFG(temporalGradient, opticalFlow, spatialGradient, os.path.join(resultsPath, f"VFG_{eventIndexCounter}.png"),True)
        create_VIGF(
            temporalGradient,
            intensity,
            spatialGradient,
            opticalFlow,
            vigfPath,
            save=True,
            cutoff=0.1)
        temporalGradient = np.zeros_like(temporalGradient)
        # temporalGradient = exponentialDecay(temporalGradient, decayTimeSurface, event[0], 0.0, settings["decayParameter"])
        intensity = exponentialDecay(intensity, decayTimeSurface, event[0], settings["neutralPotential"], settings["decayParameter"])
        # spatialGradient = exponentialDecay(spatialGradient, decayTimeSurface, event[0], 0.0, settings["decayParameter"]*100)
        spatialGradient = np.zeros_like(spatialGradient)
        # opticalFlow = np.zeros_like(opticalFlow)

        # opticalFlow = exponentialDecay(opticalFlow, decayTimeSurface, event[0], 0.0, settings["decayParameter"]*100)
        # opticalFlow = np.random.uniform(-1,1,(*opticalFlow.shape,))
        eventIndexCounter += 1
        


















    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds")