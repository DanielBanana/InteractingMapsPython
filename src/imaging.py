import cv2
import numpy as np
from fileOperations import loadCalibration

def undisortFrame(frame, cameraMatrix, distortionParameters):
    return cv2.undistort(frame, cameraMatrix=cameraMatrix, distCoeffs=distortionParameters)

def undistortFrames(frames: np.ndarray, cameraMatrix, distortionParameters):
    return np.array(list(map(lambda f,c,d: cv2.undistort(f,c,d), frames, cameraMatrix, distortionParameters)))

def intensit2Image(frame: cv2.typing.MatLike):
    maxPolarity = 255.0
    minPolarity = 0.0
    frame = np.float32(frame)
    frame = frame * 1.0 / (maxPolarity - minPolarity) - minPolarity / (maxPolarity - minPolarity)
    grey = np.uint8(frame * 255)
    return grey

def temporalGradient2Image(temporalGradient, cutoff=0.1):
    image = np.zeros((*temporalGradient.shape,3), np.int8)
    on = np.where(temporalGradient > cutoff, 255, 0)
    off = np.where(temporalGradient < -cutoff, 255, 0)
    image[:,:,1] = on
    image[:,:,2] = off
    return image

def vectorField2Image(vectorField: np.ndarray):
    rows, cols, _ = vectorField.shape

    # Extract y and x components from the vector field
    y = vectorField[..., 0]  # All rows, all cols, 0th channel (y)
    x = vectorField[..., 1]  # All rows, all cols, 1st channel (x)

    # Compute angles using arctan2 (handles all quadrants)
    angles = np.arctan2(y, x)

    # Compute saturations (magnitude of the vector)
    saturations = np.sqrt(x**2 + y**2)

    # Normalize angles to [0, 179] (OpenCV HSV hue range)
    # hue = ((angles + np.pi) / (2 * np.pi)) * 179
    angles_positive = np.where(angles < 0, angles + 2 * np.pi, angles)
    hue = angles_positive/(2*np.pi) * 179
    hue = hue.astype(np.uint8)

    # Value channel (full brightness)
    value = np.full_like(hue, 255, dtype=np.uint8)

    # Normalize saturations to [0, 255]
    max_saturation = np.max(saturations)
    min_saturation = np.min(saturations)
    saturation = np.zeros_like(hue, dtype=np.uint8)

    # Avoid division by zero and handle zero saturation
    mask = saturations > 0
    saturation[mask] = np.clip(
        (saturations[mask] / max_saturation) * 255,
        100,  # Minimum saturation (as in your C++ code)
        255
    ).astype(np.uint8)
    value[~mask] = 0  # Set value to 0 where saturation is 0

    # Merge HSV channels
    hsv_image = cv2.merge([hue, saturation, value])

    # Convert HSV to BGR
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return bgr_image

def create_outward_vector_field(grid_size):
    # Create 1D arrays for x and y coordinates
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(1.0, -1.0, grid_size)

    # Create 2D meshgrid
    xv, yv = np.meshgrid(x, y, indexing='xy')

    # Initialize 3D array for vector field
    vectors = np.zeros((grid_size, grid_size, 2), dtype=np.float32)

    # Calculate normalized vectors
    distance = np.sqrt(xv**2 + yv**2)
    distance[distance == 0] = 1.0  # Avoid division by zero

    vectors[..., 0] = yv / distance  # y-component
    vectors[..., 1] = xv / distance  # x-component

    return vectors

def create_circular_band_mask(image_size, inner_radius, outer_radius):
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    center_x, center_y = width // 2, height // 2

    # Create grid of coordinates
    j, i = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate distances from center
    distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)

    # Apply band condition
    mask[(distance >= inner_radius) & (distance <= outer_radius)] = 255

    return mask


def create_outward_vector_field(grid_size):
    # Create 1D arrays for x and y coordinates
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(1.0, -1.0, grid_size)

    # Create 2D meshgrid
    xv, yv = np.meshgrid(x, y, indexing='xy')

    # Initialize 3D array for vector field
    vectors = np.zeros((grid_size, grid_size, 2), dtype=np.float32)

    # Calculate normalized vectors
    distance = np.sqrt(xv**2 + yv**2)
    distance[distance == 0] = 1.0  # Avoid division by zero

    vectors[..., 0] = yv / distance  # y-component
    vectors[..., 1] = xv / distance  # x-component

    return vectors

def create_circular_band_mask(image_size, inner_radius, outer_radius):
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    center_x, center_y = width // 2, height // 2

    # Create grid of coordinates
    j, i = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate distances from center
    distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)

    # Apply band condition
    mask[(distance >= inner_radius) & (distance <= outer_radius)] = 255

    return mask


def create_colorwheel(grid_size):
    # Create outward vector field
    vector_field = create_outward_vector_field(grid_size)

    # Convert vector field to BGR image (assuming vector_field2image is implemented)
    image = vectorField2Image(vector_field)

    # Define inner and outer radius
    inner_radius = grid_size / 4.0
    outer_radius = grid_size / 2.0

    # Create circular band mask
    mask = create_circular_band_mask(image.shape[:2], inner_radius, outer_radius)

    # Apply mask to the image
    colourwheel = image.copy()
    colourwheel[mask == 0] = [255, 255, 255]  # Set outside mask to white

    return colourwheel


def create_VIGF(V: np.ndarray, I: np.ndarray, G: np.ndarray, F: np.ndarray, path="VIGF.png", save=False, cutoff=0.1):
    # Convert input matrices to images
    V_img = temporalGradient2Image(V, cutoff)  # Assume V2image is implemented
    I_img = intensit2Image(I)  # Assume frame2grayscale is implemented
    G_img = vectorField2Image(G)  # Assume vector_field2image is implemented
    F_img = vectorField2Image(F)

    # Apply mask (if needed, currently skipped as in the original code)
    masked_F = F_img  # Default: no masking

    # Get dimensions
    rows, cols = V.shape
    I_rows, I_cols = I.shape

    # Calculate color wheel size and output image size
    colourwheel_size = cols // 2
    y_size = rows * 2 + 20
    x_size = cols * 2 + 30 + colourwheel_size

    # Create blank output image
    image = np.zeros((y_size, x_size, 3), dtype=np.uint8)

    # Place V image
    image[5:5+rows, 5:5+cols] = V_img

    # Place I image (convert to BGR)
    I_img_bgr = cv2.cvtColor(I_img, cv2.COLOR_GRAY2BGR)
    image[5:5+I_rows, 5+cols+15+colourwheel_size:5+cols+15+colourwheel_size+I_cols] = I_img_bgr

    # Place G image
    image[rows+15:rows+15+rows, 5:5+cols] = G_img

    # Place F image
    image[rows+15:rows+15+rows, 5+cols+15+colourwheel_size:5+cols+15+colourwheel_size+cols] = masked_F

    # Create and place the color wheel
    colourwheel = create_colorwheel(colourwheel_size)  # Assume create_colorwheel is implemented
    image[2*rows+10-colourwheel_size:2*rows+10, cols+10:cols+10+colourwheel_size] = colourwheel

    # Save the image if required
    if save:
        cv2.imwrite(path, image)

    return image

def createColorbar(globalMin, globalMax, height, width, colormapType=cv2.COLORMAP_VIRIDIS):
    """
    Create a colorbar image with min/max labels.

    Args:
        globalMin: Minimum value for the colorbar
        globalMax: Maximum value for the colorbar
        height: Height of the colorbar
        width: Width of the colorbar
        colormapType: OpenCV colormap type
    """
    # Create vertical gradient
    colorbar = np.linspace(globalMin, globalMax, height, dtype=np.float32)
    colorbar = np.tile(colorbar[:, np.newaxis], (1, width))

    # Normalize to [0, 255]
    colorbarNormalized = ((colorbar - globalMin) / (globalMax - globalMin) * 255).astype(np.uint8)

    # Apply colormap
    colorbarColored = cv2.applyColorMap(colorbarNormalized, colormapType)

    # Add labels
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    thickness = 1
    color = (255, 255, 255)  # White

    # Min label (bottom)
    minLabel = f"{globalMin:.2f}"
    cv2.putText(colorbarColored, minLabel, (5, height - 5), fontFace, fontScale, color, thickness)

    # Max label (top)
    maxLabel = f"{globalMax:.2f}"
    cv2.putText(colorbarColored, maxLabel, (5, 20), fontFace, fontScale, color, thickness)

    return colorbarColored

def plot_VvsFG(V: np.ndarray, F: np.ndarray, G: np.ndarray, path="VvsFG.png", save=False):
    """
    Plot V, F·G, and their difference as a colored image with a colorbar.

    Args:
        V: 2D numpy array (event information)
        F: 3D numpy array (optical flow)
        G: 3D numpy array (spatial gradient)
        path: Output path for the image
        save: If True, save to disk; else, display
    """
    rows, cols = V.shape

    # Step 1: Prepare matrices
    V_img = V  # Assume already in correct format
    FdotG = -np.einsum("ijk,ijk->ij", F, G)
    diff = V - FdotG

    # Step 2: Find global min/max
    globalMin = np.min(V_img)
    globalMax = np.max(V_img)

    # Step 3: Normalize to [0, 255]
    def normalize(arr):
        return ((arr - globalMin) / (globalMax - globalMin) * 255).astype(np.uint8)

    normalizedMatrix1 = normalize(V_img)
    normalizedMatrix2 = normalize(FdotG)
    normalizedMatrix3 = normalize(diff)

    # Step 4: Apply colormap
    coloredMatrix1 = cv2.applyColorMap(normalizedMatrix1, cv2.COLORMAP_JET)
    coloredMatrix2 = cv2.applyColorMap(normalizedMatrix2, cv2.COLORMAP_JET)
    coloredMatrix3 = cv2.applyColorMap(normalizedMatrix3, cv2.COLORMAP_JET)

    # Step 5: Concatenate horizontally
    combinedMatrix = np.hstack([coloredMatrix1, coloredMatrix2, coloredMatrix3])

    # Step 6: Create colorbar
    colorbar = createColorbar(globalMin, globalMax, combinedMatrix.shape[0], 50, colormapType=cv2.COLORMAP_JET)

    # Step 7: Concatenate colorbar
    coloredImage = np.hstack([combinedMatrix, colorbar])

    # Step 8: Add title
    titleHeight = 50
    titleImage = np.zeros((titleHeight, coloredImage.shape[1], 3), dtype=np.uint8)
    title = "V | - F dot G | Difference"
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 2
    color = (255, 255, 255)  # White

    # Center title
    textSize = cv2.getTextSize(title, fontFace, fontScale, thickness)[0]
    textOrg = ((titleImage.shape[1] - textSize[0]) // 2, (titleHeight + textSize[1]) // 2)
    cv2.putText(titleImage, title, textOrg, fontFace, fontScale, color, thickness)

    # Step 9: Concatenate title
    finalImage = np.vstack([titleImage, coloredImage])

    # Step 10: Save or display
    if save:
        cv2.imwrite(path, finalImage)
    else:
        cv2.imshow("V vs F dot G", finalImage)
        cv2.waitKey(0)

    return finalImage



def saveImage(Image, path="Image.png", Imode=True):
    """
    Save a 2D or 3D numpy array as an image.

    Args:
        Image: 2D or 3D numpy array
        path: Output path for the image
        Imode: If True, treat as grayscale; else, use V2image
    """
    if Image.ndim == 2:
        if Imode:
            grayImage = intensit2Image(Image)  # Assume implemented
            grayImage = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR)
        else:
            grayImage = temporalGradient2Image(Image, cutoff=0.1)  # Assume implemented
        cv2.imwrite(path, grayImage)
    elif Image.ndim == 3:
        grayImage = vectorField2Image(Image)  # Assume implemented
        cv2.imwrite(path, grayImage)
    else:
        raise ValueError("Unsupported array dimensions")


if __name__ == "__main__":
    frame = cv2.imread("Kabel 1.jpeg")
    height, width, d = frame.shape
    calibrationData = loadCalibration("res/shapes_rotation/calib.json", height, width)

    undistort = undisortFrame(frame,
                          cameraMatrix=calibrationData.cameraMatrix,
                          distortionParameters=calibrationData.distortionCoeffiecents)
    
    # grey = frame2grey(undistort)

    vectorField = np.zeros((height, width, 2))
    vectorField[...,1] = -1.0
    vecImage = vectorField2Image(vectorField)

    grid_size = 256  # Test with a 256x256 grid

    # 1. Test create_outward_vector_field
    vector_field = create_outward_vector_field(grid_size)
    print("Vector field shape:", vector_field.shape)

    # 2. Test vector_field2image (helper)
    bgr_image = vectorField2Image(vector_field)
    cv2.imwrite("vector_field_visualization.png", bgr_image)

    # 3. Test create_circular_band_mask
    inner_radius = grid_size // 4
    outer_radius = grid_size // 2
    mask = create_circular_band_mask((grid_size, grid_size), inner_radius, outer_radius)
    cv2.imwrite("circular_band_mask.png", mask)

    # 4. Test create_colorwheel
    colorwheel = create_colorwheel(grid_size)
    cv2.imwrite("colorwheel.png", colorwheel)

    print("Results saved to disk.")

    cv2.imwrite("vec.png", vecImage)

    # --- Test create_VIGF ---
    print("Testing create_VIGF...")

    # Generate synthetic data
    rows, cols = 128, 128
    V = np.random.rand(rows, cols).astype(np.float32)  # Event information
    I = np.random.rand(rows, cols).astype(np.float32)  # Grayscale image
    G = np.random.rand(rows, cols, 2).astype(np.float32)  # Spatial gradient
    F = np.random.rand(rows, cols, 2).astype(np.float32)  # Optical flow

    # Call create_VIGF
    vigf_image = create_VIGF(V, I, G, F, path="test_VIGF.png", save=True)
    print("VIGF image shape:", vigf_image.shape)
    print("Saved VIGF image to 'test_VIGF.png'")

    # --- Test plot_VvsFG ---
    print("\nTesting plot_VvsFG...")

    # Generate synthetic data
    V = np.random.rand(rows, cols).astype(np.float32)  # Event information
    F = np.random.rand(rows, cols, 2).astype(np.float32)  # Optical flow
    G = np.random.rand(rows, cols, 2).astype(np.float32)  # Spatial gradient

    # Call plot_VvsFG
    plot_image = plot_VvsFG(V, F, G, path="test_plot_VvsFG.png", save=True)
    print("Plot image shape:", plot_image.shape)
    print("Saved plot image to 'test_plot_VvsFG.png'")

    # --- Optional: Display images ---
    # Uncomment to display images (requires a GUI environment)
    # cv2.imshow("VIGF", vigf_image)
    # cv2.imshow("Plot", plot_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()