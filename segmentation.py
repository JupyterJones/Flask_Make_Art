#https://www.threads.net/@vivllainous
import cv2
import numpy as np

def remove_small_objects(mask, min_size):
  # Find connected components
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
  sizes = stats[:, -1]
  mask = np.zeros_like(mask)
  for i in range(1, num_labels):
    if sizes[i] >= min_size:
      mask[labels == i] = 255
  return mask

# Load mask image
mask = cv2.imread('static/archived-images/inverted_binary_mask.png', 0)

# Remove small objects
cleaned_mask = remove_small_objects(mask, 200)  # Adjust min_size as needed

cv2.imwrite('static/archived-images/segmented-mask.png', cleaned_mask)

