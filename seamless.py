from PIL import Image, ImageFilter
import glob
import random
import logging
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def feather_image(image, radius=50):
    """Applies a feathered transparency effect to the left and right edges of an image."""
    logging.info(f"Applying feather effect with radius {radius} to image of size {image.size}")
    
    # Create an alpha mask with the same size as the image
    mask = Image.new("L", image.size, 0)
    
    # Apply feathering to the left and right edges
    mask.paste(255, (radius, 0, image.width - radius, image.height))
    mask = mask.filter(ImageFilter.GaussianBlur(radius))
    
    # Apply the mask to the image
    image.putalpha(mask)
    return image

def create_seamless_image(images, feather_radius=5, overlap=100):
    """Creates a seamless image by blending the provided images with feathered edges and overlap."""
    total_width = sum(img.width for img in images) - overlap * (len(images) - 1)
    max_height = max(img.height for img in images)

    logging.info(f"Creating combined image of size {total_width}x{max_height}")
    
    # Create a new image with the total width and max height
    combined_image = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for i, img in enumerate(images):
        feathered_img = feather_image(img, feather_radius)
        combined_image.paste(feathered_img, (x_offset, 0), feathered_img)
        x_offset += img.width - overlap  # Overlap the images to ensure they blend seamlessly
        logging.info(f"Image {i+1} pasted at position {x_offset}")

    return combined_image

# Load your images
image_files = random.sample(glob.glob('static/archived-store-bak/*.png'), 8)

if len(image_files) < 8:
    logging.warning("Less than 5 images found. Adjusting the number of selected images.")

# Resize images to ensure consistency
images = [Image.open(img).convert('RGBA').resize((512, 768), resample=Image.LANCZOS) for img in image_files]

# Create the seamless image
seamless_image = create_seamless_image(images, feather_radius=10, overlap=100)
output_path = 'static/seamless_image.png'
seamless_image.save(output_path)
logging.info(f"Seamless image saved as {output_path}")

from moviepy.editor import ImageClip, VideoClip
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_scrolling_video(image_path, output_video_path, video_duration=10, video_size=(512, 768)):
    """Creates a video by scrolling across the image from left to right."""
    
    logging.info(f"Loading image from {image_path}")
    
    # Load the image
    image = ImageClip(image_path)

    def scroll_func(get_frame, t):
        """Defines the scrolling effect by moving the image horizontally."""
        x = int((image.size[0] - video_size[0]) * t / video_duration)
        return get_frame(t)[0:video_size[1], x:x+video_size[0]]
    
    # Create the video clip with the scrolling effect
    video = VideoClip(lambda t: scroll_func(image.get_frame, t), duration=video_duration)
    
    # Set the frames per second
    video = video.set_fps(24)

    # Write the video file
    logging.info(f"Saving video to {output_video_path}")
    video.write_videofile(output_video_path, codec='libx264', audio=False)

# Define the paths and parameters
image_path = 'static/seamless_image.png'
output_video_path = 'static/seamless_video.mp4'
video_duration = 34  # duration of the video in seconds
video_size = (512, 768)  # size of the output video

# Create the scrolling video
make_scrolling_video(image_path, output_video_path, video_duration, video_size)

subprocess.run(["add_frameL", output_video_path])
