import sys
import os
import jax.numpy as jnp
import logging
import json
from pathlib import Path
from datatypes import CalibrationData

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

def writeToFile(tensor: jnp.array, path):
    split = str.split(path, ".")
    if split[-1] == ".txt":
        jnp.save
    elif split[-1] == ".npy":
        jnp.save(file=path, arr=tensor)
    else:
        logger.info("File extension missing or not recognised. Use .txt or .npy")

from pathlib import Path

def createFolderAndUpdateGitIgnore(folder_name: str) -> Path:
    """
    Creates a folder at current_directory/output/folder_name and adds the folder to .gitignore
    to prevent git bloat.

    Args:
        folder_name: Name of the folder to create under output/

    Returns:
        Path: Absolute path to the created folder
    """
    current_directory = Path.cwd()
    output_folder_path = current_directory / "output"

    # Create output folder if it doesn't exist
    output_folder_path.mkdir(exist_ok=True)

    # Create the target folder
    folder_path = output_folder_path / folder_name
    folder_path.mkdir(exist_ok=True)

    # Path to .gitignore
    gitignore_path = current_directory / ".gitignore"

    # Check if folder is already in .gitignore
    folder_in_gitignore = False
    if gitignore_path.exists():
        with open(gitignore_path, "r") as gitignore_file:
            for line in gitignore_file:
                line = line.strip()
                if line == folder_name or line == f"/{folder_name}":
                    folder_in_gitignore = True
                    break

    # Add folder to .gitignore if not already present
    if not folder_in_gitignore:
        with open(gitignore_path, "a") as gitignore_file:
            gitignore_file.write(f"\n{folder_name}\n")

    return folder_path


def loadSettings(jsonPath: str) -> dict:
    """
    Load settings from a JSON file.

    Args:
        jsonPath: Path to the JSON file.

    Returns:
        dict: Dictionary containing the settings.
    """
    with open(jsonPath, "r") as file:
        settings = json.load(file)

    # Calculate derived values
    settings["fps"] = 1.0 / settings["timeStep"]
    settings["FR_updates_per_second"] = 1.0 / settings["timeStep"]

    return settings

def loadCalibration(jsonPath: str, height: int, width: int):
    """
    Load calibration settings from a JSON file.

    Args:
        jsonPath: Path to the JSON file.

    Returns:
        dict: Dictionary containing the settings.
    """
    with open(jsonPath, "r") as file:
        settings = json.load(file)
    focalPoint = settings.focalPoint
    opticalCenter = settings.opticalCenter
    distortionCoefficients = settings.distortionCoefficients
    cameraMatrix = jnp.array([
        [focalPoint[0], 0, opticalCenter[0]],
        [0, focalPoint[1], opticalCenter[1]],
        [0, 0, 1]
    ])
    viewAngles = jnp.array([2*jnp.atan(height/(2 * focalPoint[0])),
                            2*jnp.atan(width/(2*focalPoint[1]))])
    calibrationData  = CalibrationData(focalPoint=settings.focalPoint,
                                       cameraMatrix=cameraMatrix,
                                       distortionCoefficients=distortionCoefficients,
                                       viewAngles=viewAngles)
    return calibrationData

