#!/home/jack/miniconda3/envs/cloned_base/bin/python
import os
import subprocess
from PIL import Image
import glob
import random
import imageio
import numpy as np
from sys import argv
import shutil
import uuid

# Get the image directory from command-line arguments
image_directory = argv[1]

# Set the output directory for generated videos

output_directory = os.path.join(image_directory, "output_videos/")
#if not os.path.exists(output_directory):
#    os.makedirs(output_directory)
print("image_directory:", image_directory)
print("output_directory:", output_directory)

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get the list of images in the directory and shuffle them
image_list = sorted(glob.glob(os.path.join(image_directory, '*.jpg')))
random.shuffle(image_list)
print("Number of images:", len(image_list))

# Initialize a list to store the paths of generated videos
video_paths = []

# Iterate through consecutive pairs of images
for i in range(len(image_list) - 1):
    base_image_path = image_list[i]
    image_to_paste_path = image_list[i]# + 1]
    print("base_image_path:", base_image_path)
    print("image_to_paste_path:", image_to_paste_path)

    # Open the base image
    base_image = Image.open(base_image_path).convert("RGBA")
    bs = base_image.size
    print("Base image size:", bs)

    # Create a list to store individual frames
    IMG_SRC = []

    # Open each image to paste and create frames
    # Iterate through each row of the base image
    #for j in range(0, bs[0], 15):
    for j in range(0, bs[0], 5):
        current_frame = base_image.copy()
        image_to_paste = Image.open(image_to_paste_path).convert("RGBA")
        print("Image to paste size:", image_to_paste.size)
        image_to_paste = image_to_paste.resize((bs[0] - j, bs[1]), Image.BICUBIC)

        # Determine the position where you want to paste the smaller image on the larger image
        x = 0
        y = 0
        paste_position = (x, y)

        # Ensure that the smaller image is not larger than the base image
        if image_to_paste.size[0] + paste_position[0] <= base_image.size[0] and \
                image_to_paste.size[1] + paste_position[1] <= base_image.size[1]:
            # Paste the smaller image onto the larger image
            current_frame.paste(image_to_paste, paste_position, image_to_paste)

            # Append the current frame to the list
            IMG_SRC.append(np.array(current_frame))

    # Save the frames as an MP4 video using imageio
    output_video_path = os.path.join(output_directory, f'output_video_{i}.mp4')
    print("output_video_path:", output_video_path)
    imageio.mimsave(output_video_path, IMG_SRC, fps=30)
    video_paths.append(output_video_path)

# Prepare for concatenation of all generated videos
input_list_path = os.path.join(output_directory, "input_list.txt")
with open(input_list_path, 'w') as input_list_file:
    for video_path in video_paths:
        input_list_file.write(f"file '{video_path}'\n")

# Concatenate videos using ffmpeg
concatenated_video_path = os.path.join(output_directory, "final_result.mp4")
ffmpeg_command = f"ffmpeg -y -f concat -safe 0 -i {input_list_path} -c copy {concatenated_video_path}"
subprocess.run(ffmpeg_command, shell=True)

# Use uuid to create a unique name for the output video and copy it to the final directory
final_output_directory = "/home/jack/Desktop/HDD500/collections/vids/"
os.makedirs(final_output_directory, exist_ok=True)
final_output_video_path = os.path.join(final_output_directory, str(uuid.uuid4()) + ".mp4")
shutil.copyfile(concatenated_video_path, final_output_video_path)

print(f"Final video saved to {final_output_video_path}")