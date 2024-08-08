import cv2
from random import randint
from PIL import Image
import time
import random
import os
import glob
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories
junk_dir = 'static/junk'
archived_images_dir = 'static/archived-images'
videos_dir = '/mnt/HDD500/Image_Retriever/static/videos'

def setup_directories():
    """Create necessary directories."""
    logging.info("Setting up directories...")

    if os.path.exists(junk_dir):
        logging.info(f"Removing existing directory: {junk_dir}")
        shutil.rmtree(junk_dir)
    os.makedirs(junk_dir)
    logging.info(f"Created directory: {junk_dir}")

    if os.path.exists(archived_images_dir):
        logging.info(f"Removing existing directory: {archived_images_dir}")
        shutil.rmtree(archived_images_dir)
    os.makedirs(archived_images_dir)
    logging.info(f"Created directory: {archived_images_dir}")

def vid2img(count, videofile, outputpath):
    """Extract a random frame from a video and save it as an image."""
    logging.info(f"Processing video file: {videofile}")

    vidcap = cv2.VideoCapture(videofile)
    totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    randomFrameNumber = randint(0, totalFrames - 1)
    logging.debug(f"Total frames: {totalFrames}, Selected frame: {randomFrameNumber}")

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, randomFrameNumber)
    success, image = vidcap.read()

    if success:
        logging.info(f"Frame extraction successful: {randomFrameNumber}")
        temp_image_path = os.path.join(junk_dir, 'archived-images.jpg')
        cv2.imwrite(temp_image_path, image)
        IM = Image.open(temp_image_path)
        im = IM.resize((512, 768))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(outputpath, f"Frame{randomFrameNumber}-{count}_{timestr}_.jpg")
        im.save(filename)
        logging.info(f"Saved image: {filename}")
        return Image.open(filename)
    else:
        logging.error("Failed to extract frame from video")
        return None

def get_frame():
    """Process multiple video files to extract frames."""
    setup_directories()
    vid_list = glob.glob(os.path.join(videos_dir, '*.mp4'))
    random.shuffle(vid_list)

    for i in range(min(25, len(vid_list))):
        logging.info(f"Processing video {i + 1}/{len(vid_list)}")
        vid2img(i, vid_list[i], archived_images_dir)
        logging.info(f"Video {i + 1} done")
        time.sleep(1)

if __name__ == "__main__":
    get_frame()
    logging.info("All done")
