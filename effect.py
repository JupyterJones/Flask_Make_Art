import moviepy.editor as mp
import numpy as np
import cv2
from sys import argv
def apply_vintage_effect(image):
    # Convert image to sepia (vintage-like effect)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255)  # Ensure pixel values are in valid range

    # Add random noise to simulate grain
    noise = np.random.normal(loc=0, scale=25, size=image.shape)  # Gaussian noise
    noisy_image = sepia_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are in valid range
    
    # Optionally, simulate scratches (can adjust for random or constant placement)
    scratch_width = 3
    for i in range(0, image.shape[1], 200):  # Add vertical scratches every 200 pixels
        noisy_image[:, i:i+scratch_width] = 255  # White scratch lines
    
    return noisy_image.astype(np.uint8)

def vintage_effect_on_video(input_path, output_path):
    # Load the video
    clip = mp.VideoFileClip(input_path)
    
    # Apply the vintage effect to each frame
    vintage_clip = clip.fl_image(apply_vintage_effect)
    
    # Write the result to a new file
    vintage_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

# Apply the effect to your video
input_video = argv[1]
output_video = "output_vintage_video.mp4"
vintage_effect_on_video(input_video, output_video)

