import dlib
import cv2
import numpy as np
import os
from  PIL import Image

def create_feathered_image(foreground_path, output_path):
    # Load the foreground image
    foreground = cv2.imread(foreground_path)
    height, width = foreground.shape[:2]

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_foreground)

    # Check if any face is detected
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Create an alpha channel and a binary mask
    alpha_channel = np.zeros((height, width), dtype=np.uint8)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        cv2.circle(alpha_channel, center, radius, 255, -1)

    # Feather the edges of the mask
    mask = cv2.GaussianBlur(alpha_channel, (101, 101), 0)

    # Add the alpha channel to the foreground image
    foreground_rgba = np.dstack((foreground, mask))

    # Save the result as a PNG file with transparency
    cv2.imwrite(output_path, foreground_rgba)

    print(f"Feathered image saved to: {output_path}")

    return output_path

def overlay_feathered_on_background(foreground_path, background_path, output_path):
    # Load the feathered image and background image
    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
    background = cv2.imread(background_path)

    # Resize and crop both images to 512x768
    foreground = resize_and_crop(foreground)
    background = resize_and_crop(background)

    # Extract the alpha channel from the foreground image
    alpha_channel = foreground[:, :, 3] / 255.0
    foreground_rgb = foreground[:, :, :3]

    # Ensure background has 4 channels
    background_rgba = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    background_alpha = background_rgba[:, :, 3] / 255.0

    # Blend the images
    for i in range(3):  # For each color channel
        background_rgba[:, :, i] = (foreground_rgb[:, :, i] * alpha_channel + background_alpha * background_rgba[:, :, i] * (1 - alpha_channel)).astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, background_rgba)

    print(f"Composite image saved to: {output_path}")
    
    im = Image.open(output_path).convert('RGB')
    im.save(output_path[:-3]+'jpg', quality=95)    
    return output_path

def resize_and_crop(image, target_width=512, target_height=768):
    # Resize the image to fit the target dimensions while maintaining the aspect ratio
    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    # Crop the resized image to the target dimensions
    crop_x = (new_width - target_width) // 2
    crop_y = (new_height - target_height) // 2
    cropped_image = resized_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

    return cropped_image

# Example usage
foreground_image_path = 'static/archived-images/Frame7-9_20240810-065130_.jpg'
feathered_image_path = 'static/archived-images/feathered_face.png'
background_image_path = 'static/archived-images/Frame9-8_20240810-065129_.jpg'
output_composite_path = 'static/archived-images/composite_image.png'

# Create a feathered PNG image from the detected face
create_feathered_image(foreground_image_path, feathered_image_path)

# Overlay the feathered image on the background
overlay_feathered_on_background(feathered_image_path, background_image_path, output_composite_path)
