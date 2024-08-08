import glob
import random
from PIL import Image
import os
import logging
import subprocess
import uuid
import shutil

from moviepy.video.compositing.transitions import slide_in
from moviepy.video.fx import all
from moviepy.editor import *
import glob
import random
from PIL import Image
import cv2
import os
import uuid
import shutil
from sys import argv

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def prep_homedirectory():
    image_directory = 'static/temp_images_exp'
    #image_directory = 'static/KLING'
    logging.info(f"Image directory: {image_directory}")

    # Create or clear the image directory
    if os.path.exists(image_directory):
        shutil.rmtree(image_directory)
        logging.info(f"Cleared contents of image directory: {image_directory}")
    os.makedirs(image_directory, exist_ok=True)
    logging.info(f"Created image directory: {image_directory}")

    # Copy all jpg files from source to image_directory
    for f in os.listdir('static/archived-store'):
        #ends in jpg or png
        if f.endswith('.jpg') or f.endswith('.png'):
            logging.info(f"Copying {f} to {image_directory}")
            shutil.copy(os.path.join('static','archived-store', f), image_directory)

    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]    
    logging.info(f"Image files: {image_files}")

    # Shuffle the list of image files
    random.shuffle(image_files)
    logging.info(f"Shuffled image files: {image_files}")

    return image_files

def image_dir_to_zoom():
    selected_directory = 'static/temp_images_exp'
    if not os.path.exists(selected_directory):
        #make it
        os.makedirs(selected_directory)        
    try:
        image_files = glob.glob(f'{selected_directory}/*.png')
        if not image_files:
            logging.error("No images found in the directory.")
            return

        SIZE = Image.open(random.choice(image_files)).size
    except Exception as e:
        logging.error(f"Error opening image: {e}")
        return

    output_video = 'generated_video_exp.mp4'
    frame_rate = 60  # Adjust the frame rate as needed
    zoom_increment = 0.0005
    zoom_duration = 300
    width, height = SIZE

    ffmpeg_cmd = (
        f"ffmpeg -hide_banner -pattern_type glob -framerate {frame_rate} "
        f"-i '{selected_directory}/*.png' "
        f"-vf \"scale=8000:-1,zoompan=z='min(zoom+{zoom_increment},1.5)':x='iw/2':y='ih/2-4000':d={zoom_duration}:s={width}x{height},crop={width}:{height}:0:256\" "
        f"-c:v libx264 -pix_fmt yuv420p -r {frame_rate} -s {width}x{height} -y {output_video}"
    )

    logging.info(f"FFmpeg command: {ffmpeg_cmd}")
    try:
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
        logging.info("Video generated successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg command failed: {e}")

    # Generate a unique name for the video file and copy it to the assets directory
    video_name = str(uuid.uuid4()) + '_zoom_exp.mp4'
    if not os.path.exists('static/assets_exp'):
        os.makedirs('static/assets_exp')    
    shutil.copy(output_video, os.path.join('static/assets_exp', video_name))

    output_vid = os.path.join('static/assets_exp', video_name)
    logging.info(f"Generated video: {output_vid}")
    return output_vid

def add_title_image(video_path, hex_color="#A52A2A"):
    # Define the directory path
    directory_path = "static/temp_exp"
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If not, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    # Load the video file and title image
    video_clip = VideoFileClip(video_path)
    print(f"Video size: {video_clip.size}")
    width, height = video_clip.size
    title_image_path = "/mnt/HDD500/EXPER/static/assets/Title_Image02.png"
    
    # Set the desired size of the padded video (e.g., video width + padding, video height + padding)
    padded_size = (width + 80, height + 80)
    
    # Calculate the position for centering the video within the larger frame
    x_position = (padded_size[0] - video_clip.size[0]) / 2
    y_position = (padded_size[1] - video_clip.size[1]) / 2

    # Convert hex color to RGB tuple
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    rgb_tuple = (r, g, b)

    # Create a background ColorClip with the specified color
    background_clip = ColorClip(padded_size, color=rgb_tuple)
    
    # Add the video clip on top of the background
    padded_video_clip = CompositeVideoClip([background_clip, video_clip.set_position((x_position, y_position))])
    padded_video_clip = padded_video_clip.set_duration(video_clip.duration)

    # Load the title image
    title_image = ImageClip(title_image_path)
    title_image = title_image.set_duration(video_clip.duration)

    # Resize and position the title image
    title_image = title_image.set_position((0, -5)).resize(padded_video_clip.size)

    # Create a composite video clip with the title image overlay
    composite_clip = CompositeVideoClip([padded_video_clip, title_image])

    # Ensure the composite clip duration matches the video clip duration
    composite_clip = composite_clip.set_duration(video_clip.duration)

    # Load a random background music
    mp3_files = glob.glob("/mnt/HDD500/collections/music_long/*.mp3")
    random.shuffle(mp3_files)
    mp_music = random.choice(mp3_files)

    # Load the background music
    music_clip = AudioFileClip(mp_music)

    # Set fade-in and fade-out durations
    fade_duration = 0.5
    music_clip = music_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)

    # Ensure the audio duration matches the video duration
    music_clip = music_clip.set_duration(video_clip.duration)

    # Set the audio of the composite clip to the background music
    composite_clip = composite_clip.set_audio(music_clip)

    # Define the output path
    output_path = 'static/temp_exp/final_output_exp.mp4'

    # Export the final video with the background music
    composite_clip.write_videofile(output_path)

    # Generate a unique ID using uuid and copy the output file
    uid = str(uuid.uuid4())
    mp4_file = f"/home/jack/Desktop/HDD500/collections/vids/AI_Creates_Composite_Video_of_Processed_AI_Generated_Images_ID_{uid}.mp4"
    shutil.copyfile(output_path, mp4_file)
    print(f"Output file saved at: {mp4_file}")

    VIDEO = output_path
    return VIDEO




if __name__ == "__main__":
    prep_homedirectory()
    video_path = image_dir_to_zoom()
    
    if video_path:
        # Speed up the video
        #output_path = 'static/temp_exp/final_output_exp.mp4'
        cmd = [
            'ffmpeg', '-i', video_path, '-vf', 'setpts=0.3*PTS', '-c:a', 'copy', '-y', 'VideoFaster_exp.mp4'
        ]
        try:
            logging.info(f"Running FFmpeg command to speed up video: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logging.info("Video sped up successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg command failed: {e}")
            exit(1)  # Exit if the FFmpeg command fails
        
    # Add the title image to the video
    video_path = 'VideoFaster_exp.mp4'
    add_title_image(video_path, hex_color="#A52A2A")