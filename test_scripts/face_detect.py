import dlib
import cv2
import numpy as np
import os

def overlay_face_with_dlib(top_image_path, bottom_image_path, output_dir='static/archived_images'):
    os.makedirs(output_dir, exist_ok=True)

    # Load the top and bottom images
    foreground = cv2.imread(top_image_path)
    background = cv2.imread(bottom_image_path)

    # Resize and crop both images to 512x768
    foreground = resize_and_crop(foreground)
    background = resize_and_crop(background)

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Detect faces in the foreground image
    gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_foreground)

    # Initialize the mask and inverse mask
    mask = np.zeros(foreground.shape[:2], dtype="uint8")
    inverse_mask = np.ones(foreground.shape[:2], dtype="uint8") * 255

    # If a face is detected, create the mask
    if len(faces) > 0:
        for face in faces:
            # Get the coordinates of the detected face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Create a circular mask around the detected face
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(mask, center, radius, 255, -1)

            # Feather the edges of the mask
            mask = cv2.GaussianBlur(mask, (21, 21), 0)

            # Create the inverse mask for background blending
            inverse_mask = cv2.bitwise_not(mask)

    # Apply the masks and combine the images as before
    foreground_face = cv2.bitwise_and(foreground, foreground, mask=mask)
    background_masked = cv2.bitwise_and(background, background, mask=inverse_mask)
    result = cv2.add(foreground_face, background_masked)

    # Save the result
    output_filename = f"composite_image_{os.path.basename(top_image_path).split('.')[0]}_on_{os.path.basename(bottom_image_path).split('.')[0]}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, result)

    print(f"Composite image saved to: {output_path}")
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
top_image_path = 'static/archived-images/face.jpg'
bottom_image_path = 'static/archived-images/background.jpg'

# Perform the overlay with dlib face detection
overlay_face_with_dlib(top_image_path, bottom_image_path, output_dir='static/archived-images')
