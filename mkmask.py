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

    # Create torn edge patterns
    height, width, _ = img.shape

    # Create random torn patterns for each edge
    top_torn = np.random.randint(0, tear_depth, size=(width,))
    bottom_torn = np.random.randint(0, tear_depth, size=(width,))
    left_torn = np.random.randint(0, tear_depth, size=(height,))
    right_torn = np.random.randint(0, tear_depth, size=(height,))

    # Apply torn effect to top and bottom edges
    for x in range(width):
        if top_torn[x] > 0:  # Ensure there's something to tear
            img[:top_torn[x], x, 3] = 0  # Top edge
            img[:top_torn[x], x] = cv2.addWeighted(img[:top_torn[x], x], 0.5, np.zeros_like(img[:top_torn[x], x]), 0.5, 0)
        if bottom_torn[x] > 0:
            img[-bottom_torn[x]:, x, 3] = 0  # Bottom edge
            img[-bottom_torn[x]:, x] = cv2.addWeighted(img[-bottom_torn[x]:, x], 0.5, np.zeros_like(img[-bottom_torn[x]:, x]), 0.5, 0)

    # Apply torn effect to left and right edges
    for y in range(height):
        if left_torn[y] > 0:
            img[y, :left_torn[y], 3] = 0  # Left edge
            img[y, :left_torn[y]] = cv2.addWeighted(img[y, :left_torn[y]], 0.5, np.zeros_like(img[y, :left_torn[y]]), 0.5, 0)
        if right_torn[y] > 0:
            img[y, -right_torn[y]:, 3] = 0  # Right edge
            img[y, -right_torn[y]:] = cv2.addWeighted(img[y, -right_torn[y]:], 0.5, np.zeros_like(img[y, -right_torn[y]:]), 0.5, 0)

    return img

# Example usage
image_path = argv[1]
output_path = argv[2]
torn_image = create_torn_edge(image_path, tear_depth=40)
cv2.imwrite(output_path, torn_image)
