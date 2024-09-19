from moviepy.editor import *  
import os  
import glob
from PIL import Image
# Path to the directory containing images  
image_dir = 'static/archived-store/'  

# Get a list of image files in the directory  
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  

# Sort image files based on their filenames  
image_files.sort()  
print(image_files)  

# Duration for each image in seconds  
image_duration = 2  

# Initialize an empty list to hold image clips  
image_clips = []  

# Create the output directory if it doesn't exist  
output_dir = 'static/output'  
if not os.path.exists(output_dir):  
    os.makedirs(output_dir)  

# Iterate through the images to create zoom and pan transitions  
for i, image_file in enumerate(image_files):  
    image_path = os.path.join(image_dir, image_file)  
    image = ImageClip(image_path).set_duration(image_duration)  

    # Get the dimensions of the first image  
    if i == 0:  
        video_width, video_height = image.size  

    # Resize the image to match the video dimensions  
    image = image.resize((video_width, video_height))  

    # Calculate the scale factor for zooming  
    scale_factor = 1.1  # Adjust as needed  

    # Define the time intervals for the zoom and pan effect  
    zoom_duration = 0.5  # Adjust as needed  
    zoom_start_time = i * image_duration  
    zoom_end_time = zoom_start_time + zoom_duration  

    # Apply zoom in and out during the defined time interval  
    zoomed_image = image.fx(vfx.crop, x_center=0.5, y_center=0.5, width=video_width * scale_factor, height=video_height * scale_factor)  
    zoomed_image = zoomed_image.set_start(zoom_start_time).set_end(zoom_end_time)  

    # Append the zoomed image to the list of clips  
    image_clips.append(zoomed_image)  

# Concatenate all image clips into the final video  
final_video = concatenate_videoclips(image_clips)  

# Write the final video to an output file  
final_video.write_videofile(os.path.join(output_dir, 'video.mp4'), codec='libx264', fps=24)
def create_mp4_from_images(image_list, selected_directory, output_filename, fps, duration):
    image_paths = [os.path.join(image_files, image) for image in image_list]

    # Get the size of the first image
    first_image = Image.open(image_paths[0])
    width, height = first_image.size

    # Create a list of resized images as NumPy arrays
    images = [np.array(Image.open(image).resize((width, height))) for image in image_paths]

    # Create a list of clips for each image
    clips = [ImageSequenceClip([image], fps=fps).set_duration(0.5) for image in images]
    
    # Concatenate the clips into a single clip
    concat_clip = concatenate_videoclips(clips, method="compose")
    
    # Save the concatenated clip as an MP4 file
    mp4_file = "static/temp_exp/XXXtest.mp4"
    concat_clip.write_videofile(mp4_file, fps=fps)

    # Create the final clip from the original images and set the duration
    clip = ImageSequenceClip(images, fps=fps)
    duration_per_image = duration / len(images)
    final_clip = clip.set_duration(duration_per_image)

    # Write the final clip to the output file
    final_clip.write_videofile(output_filename)

# Usage example
selected_directory = "static/archived-store"
image_list = glob.glob("static/archived-store/*.jpg")
output_filename = 'static/temp_exp/outputX.mp4'
fps = 24
duration = 12  # Total duration of the video in seconds

create_mp4_from_images(image_list, selected_directory, output_filename, fps, duration)
