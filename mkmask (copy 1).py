import cv2
import numpy as np
from sys import argv

def create_torn_edge(image_path, tear_depth=40):
    """Creates a torn edge effect on an image.

    Args:
        image_path: Path to the image file.
        tear_depth: Depth of the torn edge in pixels.

    Returns:
        The image with the torn edge effect and a transparent background.
    """

    # Load the image as a NumPy array
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert to RGBA if necessary (for transparency)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    # Create masks for each edge
    height, width, channels = img.shape
    top_mask = np.zeros((tear_depth, width, channels), dtype=np.uint8)
    bottom_mask = np.zeros((tear_depth, width, channels), dtype=np.uint8)
    left_mask = np.zeros((height, tear_depth, channels), dtype=np.uint8)
    right_mask = np.zeros((height, tear_depth, channels), dtype=np.uint8)

    # Create torn edge patterns
    for mask in [top_mask, bottom_mask, left_mask, right_mask]:
        cv2.randn(mask, mean=255, stddev=50)  # Random noise for torn effect

    # Apply masks to the image
    img[:tear_depth, :, :] = cv2.addWeighted(img[:tear_depth, :, :], 0.5, top_mask, 0.5, 0)
    img[-tear_depth:, :, :] = cv2.addWeighted(img[-tear_depth:, :, :], 0.5, bottom_mask, 0.5, 0)
    img[:, :tear_depth, :] = cv2.addWeighted(img[:, :tear_depth, :], 0.5, left_mask, 0.5, 0)
    img[:, -tear_depth:, :] = cv2.addWeighted(img[:, -tear_depth:, :], 0.5, right_mask, 0.5, 0)

    # Create a transparent background
    alpha_channel = np.ones((height, width), dtype=np.uint8) * 255
    alpha_channel[:tear_depth, :] = 0
    alpha_channel[-tear_depth:, :] = 0
    alpha_channel[:, :tear_depth] = 0
    alpha_channel[:, -tear_depth:] = 0
    img[:, :, 3] = alpha_channel

    return img

# Example usage
image_path = argv[1]
output_path = argv[2]
torn_image = create_torn_edge(image_path, tear_depth=40)
cv2.imwrite(output_path, torn_image)
