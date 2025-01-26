#!/home/jack/Desktop/Flask_Make_Art/flask_env/bin/python
import os
import random
import glob
from flask import Flask, request, render_template, request, redirect, url_for, send_from_directory, send_file, flash, jsonify, make_response, Response, session, abort, send_file, after_this_request
from flask import render_template_string
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageSequence, ImageChops, ImageStat, ImageColor, ImagePalette
from moviepy.editor import ImageClip, VideoClip, clips_array, concatenate_videoclips, CompositeVideoClip, ColorClip, VideoFileClip,AudioFileClip, concatenate_audioclips, TextClip,ImageSequenceClip
#from moviepy.video.fx import resize, speedx, crop

from matplotlib import pyplot as plt
from skimage import future, data, segmentation, filters, color, io
from skimage.future import graph
import datetime
import inspect
import time
import datetime
import inspect
import subprocess
import shutil
from werkzeug.utils import secure_filename
import numpy as np
import yt_dlp
import dlib
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageFont
import glob
import subprocess
import string
import uuid
from sys import argv
from skimage.io import imread, imsave
from werkzeug.middleware.profiler import ProfilerMiddleware
app = Flask(__name__)
#app.config['PROFILE'] = True
#app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])


app.config['UPLOAD_FOLDER'] = 'static/archived-images'
app.config['TEXT_FOLDER']= 'static/TEXT'
app.config['MASK_FOLDER'] = 'static/archived-masks'
app.config['STORE_FOLDER'] = 'static/archived-store'
#app.config['STORE_FOLDER'] = 'static/KLING'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB
# Directory to save downloaded videos and extracted images
DOWNLOAD_FOLDER = 'static/downloads'
ARCHIVED_IMAGES_FOLDER = 'static/archived-images'
OVERLAY_FOLDER = 'static/overlay_zooms/title'

app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['ARCHIVED_IMAGES_FOLDER'] = ARCHIVED_IMAGES_FOLDER
TEXT_FOLDER ='static/TEXT'
RESOURCE_FOLDER = 'static/archived_resources'
VIDEO_TEMP='static/temp_exp'
# Ensure the directories exist
os.makedirs(VIDEO_TEMP, exist_ok=True)
os.makedirs(OVERLAY_FOLDER , exist_ok=True)
os.makedirs(RESOURCE_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)
os.makedirs(ARCHIVED_IMAGES_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# Configurations
app.config['TEMP_FOLDER'] = 'static/temp/'
app.config['FONT_FOLDER'] = 'static/fonts/'
app.config['PUBLISH_FOLDER'] = 'static/archived-store/'

currentDir = os.getcwd()


#@app.route('/convert', methods=['POST'])
def convert_images():
    # Directory containing the JPG images
    image_directory = 'static/archived-store'  # Adjust this path as needed
    
    logit(f"Starting conversion in directory: {image_directory}")
    
    # Check if directory exists
    if not os.path.isdir(image_directory):
        logit(f"Directory does not exist: {image_directory}")
        return redirect(url_for('index'))
    
    # Loop through all files in the directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith('.jpg'):
            # Construct full file paths
            jpg_path = os.path.join(image_directory, filename)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(image_directory, png_filename)
            
            try:
                logit(f"Converting {jpg_path} to {png_path}")
                
                # Open the JPG image and convert it to PNG
                with Image.open(jpg_path) as img:
                    img = img.convert('RGBA')  # Ensure image has alpha channel if needed
                    img.save(png_path, format='PNG')
                
                logit(f"Successfully converted {jpg_path} to {png_path}")
                
                # Remove the original JPG file
                os.remove(jpg_path)
                logit(f"Removed original file: {jpg_path}")
                
            except Exception as e:
                logit(f"Failed to convert {jpg_path}. Error: {e}")
    
    logit("Image conversion process completed.")
    return redirect(url_for('index'))
# Ensure the static/masks directory exists
if not os.path.exists('static/masks'):
    os.makedirs('static/masks')

@app.route('/mk_mask')
def mk_mask():
    masks=glob.glob('static/archived-images/mask*.jpg')
    # list by date, last image first
    masks = sorted(masks, key=os.path.getmtime, reverse=True)
    filenames = [os.path.basename(mask) for mask in masks]
    mask_data = zip(masks, filenames)
    return render_template('mk_mask.html',mask_data=mask_data)


@app.route('/create_mask', methods=['POST'])
def create_mask():
    # Get input values from the form
    x = int(request.form.get('x', 0))
    y = int(request.form.get('y', 0))
    size = int(request.form.get('size', 50)) + 20
    feather = int(request.form.get('feather', 20))
    aspect = int(request.form.get('aspect', 0))
    
    # Calculate width and height based on aspect
    if aspect > 0:
        width = size + aspect  # Make width larger for wide aspect
        height = size
    elif aspect < 0:
        width = size
        height = size + abs(aspect)  # Make height larger for tall aspect
    else:
        width, height = size, size  # Default to square aspect ratio

    # Create a black background image (size 512x768)
    background = Image.new('RGBA', (512, 768), (0, 0, 0, 255))
    
    # Create a white ellipse (or circle if width == height)
    ellipse = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ellipse)
    draw.ellipse((0, 0, width, height), fill=(255, 255, 255, 255))

    # Apply feathering (blur the edges)
    ellipse = ellipse.filter(ImageFilter.GaussianBlur(feather))

    # Calculate position to paste the ellipse (centered by default)
    paste_position = (256 + x - width // 2, 384 + y - height // 2)
    background.paste(ellipse, paste_position, ellipse)
    background = background.convert('RGB')
    
    # Optionally blur the whole background
    background = background.filter(ImageFilter.GaussianBlur(30))
    
    # Save the result in static/archived-images
    mask_path = f'static/archived-images/mask_{x}_{y}_{size}_{feather}_{aspect}.jpg'
    background.save(mask_path)
    # save a copy of the mask to static/masks with same name as the mask_path not mask png
    shutil.copy(mask_path, 'static/masks/' + os.path.basename(mask_path))
    
    
    
    # List and sort masks by date (latest first)
    masks = glob.glob('static/archived-images/mask*.jpg')
    masks = sorted(masks, key=os.path.getmtime, reverse=True)
    filenames = [os.path.basename(mask) for mask in masks]
    mask_data = zip(masks, filenames)

    logit(f"Mask saved at: {mask_path}")
    
    return render_template('mk_mask.html', mask_path=mask_path, mask_data=mask_data)





    #return send_file(mask_path, as_attachment=True)
def save_text_to_file(filename, text):
    try:
        with open(os.path.join(TEXT_FILES_DIR, filename), "w") as file:
            file.write(text)
        logit(f"File '{filename}' saved successfully.")
    except Exception as e:
        logit(f"An error occurred while saving file '{filename}': {e}")

# Route for the form
@app.route('/add_text', methods=['GET', 'POST'])
def add_text():
    # Get the list of images in the PUBLISH_FOLDER
    images = os.listdir(app.config['STORE_FOLDER'])
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]   
    
    if request.method == 'POST':
        image_file = request.form['image_file']
        text = request.form['text']
        position = (int(request.form['x_position']), int(request.form['y_position']))
        font_size = int(request.form['font_size'])
        color = request.form['color']
        font_path = os.path.join(app.config['FONT_FOLDER'], 'MerriweatherSans-Bold.ttf')
        font = ImageFont.truetype(font_path, font_size)

        logit(f"Processing image: {image_file}")
        logit(f"Text to add: '{text}' at position {position}, font size: {font_size}, color: {color}")

        # Open the image
        image_path = os.path.join(app.config['STORE_FOLDER'], image_file)
        image = Image.open(image_path)

        # Draw the text on the image
        draw = ImageDraw.Draw(image)
        draw.text(position, text, font=font, fill=color)

        # Save the temporary image for preview
        temp_image_path = os.path.join(app.config['STORE_FOLDER'], 'temp-image.png')
        image.save(temp_image_path)

        return render_template('add_text.html', images=images, selected_image=image_file, temp_image='temp-image.png', text=text, position=position, font_size=font_size, color=color)
    
    return render_template('add_text.html', images=images)


# Route to save the final image
@app.route('/save_image', methods=['POST'])
def save_image():
    image_file = request.form['image_file']
    final_text = request.form['final_text']
    position = eval(request.form['final_position'])  # Convert string back to tuple
    font_size = int(request.form['final_font_size'])
    color = request.form['final_color']
    font_path = os.path.join(app.config['FONT_FOLDER'], 'MerriweatherSans-Bold.ttf')
    font = ImageFont.truetype(font_path, font_size)

    # Open the image again
    image_path = os.path.join(app.config['PUBLISH_FOLDER'], image_file)
    image = Image.open(image_path)

    # Draw the final text on the image
    draw = ImageDraw.Draw(image)
    draw.text(position, final_text, font=font, fill=color)

    # Save the image with a unique UUID
    unique_filename = f"{uuid.uuid4()}.png"
    final_image_path = os.path.join(app.config['PUBLISH_FOLDER'], unique_filename)
    image.save(final_image_path)

    logit(f"Saved final image as: {unique_filename}")

    return redirect(url_for('add_text'))

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['MASK_FOLDER']):
    os.makedirs(app.config['MASK_FOLDER'])

if not os.path.exists(app.config['STORE_FOLDER']):
    os.makedirs(app.config['STORE_FOLDER'])

# Logging function
def logit(message):
    try:
        # Get the current timestamp
        timestr = datetime.datetime.now().strftime('%A_%b-%d-%Y_%H-%M-%S')
        print(f"timestr: {timestr}")

        # Get the caller's frame information
        caller_frame = inspect.stack()[1]
        filename = caller_frame.filename
        lineno = caller_frame.lineno

        # Convert message to string if it's a list
        if isinstance(message, list):
            message_str = ' '.join(map(str, message))
        else:
            message_str = str(message)

        # Construct the log message with filename and line number
        log_message = f"{timestr} - File: {filename}, Line: {lineno}: {message_str}\n"

        # Open the log file in append mode
        with open("static/app_log.txt", "a") as file:
            # Write the log message to the file
            file.write(log_message)

            # Print the log message to the console
            print(log_message)

    except Exception as e:
        # If an exception occurs during logging, print an error message
        print(f"Error occurred while logging: {e}")
def readlog():
    log_file_path = 'static/app_log.txt'    
    with open(log_file_path, "r") as Input:
        logdata = Input.read()
    # print last entry
    logdata = logdata.split("\n")
    return logdata

@app.route('/view_log', methods=['GET', 'POST'])
def view_log():
    data = readlog()
    return render_template('view_log.html', data=data)

def get_image_paths():
    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths.extend(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], f'*.{ext}')))
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    logit(f"Image paths: {image_paths}")
    return image_paths

def stored_image_paths():
    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths.extend(glob.glob(os.path.join(app.config['STORE_FOLDER'], f'*.{ext}')))
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    logit(f"Image paths: {image_paths}")
    return image_paths

@app.route('/')
def index():
    image_paths = stored_image_paths()
    post = get_intro(limit=1)
    decoded_post = []
    for row in post:
        # Replace newlines in the content (third field) with <br>
        id, title, content, image, video_filename = row
        if content:
            content = content.replace('\r\n', '<br>').replace('\n', '<br>')  # handle both \r\n and \n
            decoded_post.append((id, title, content, image, video_filename))
    return render_template('index.html', post=decoded_post)

@app.route('/mk_videos')
def mk_videos():
    image_paths = stored_image_paths()
    return render_template('mk_videos.html', image_paths=image_paths)
@app.route('/img_processing')
def img_processing_route():
    image_paths = stored_image_paths()
    return render_template('img_processing.html', image_paths=image_paths)
def load_images(image_directory):
    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths.extend(glob.glob(os.path.join(image_directory, f'*.{ext}')))
    #random.shuffle(image_paths)
    #image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    return image_paths[:3]
def load_image(image_directory):
    image_paths_ = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths_.extend(glob.glob(os.path.join(image_directory, f'*.{ext}')))
    #random.shuffle(image_paths)
    image_paths_ = sorted(image_paths_, key=os.path.getmtime, reverse=True)
    return image_paths_[:3]
def convert_to_grayscale(image_path):
    image = Image.open(image_path).convert('L')
    mask_path = os.path.join(app.config['MASK_FOLDER'], 'greyscale_mask.png')
    image.save(mask_path)
    #copy to upload folder
    shutil.copy(mask_path, app.config['UPLOAD_FOLDER'])
    return mask_path

def convert_to_binary(image_path):
    # Convert image to grayscale
    image = Image.open(image_path).convert('L')
    
    # Calculate the mean pixel value to use as the threshold
    np_image = np.array(image)
    threshold = np.mean(np_image)
    
    # Convert image to binary based on the mean threshold
    binary_image = image.point(lambda p: 255 if p > threshold else 0)
    
    # Save the binary mask
    mask_path = os.path.join(app.config['MASK_FOLDER'], 'binary_mask.png')
    binary_image.save(mask_path)
    
    # Invert the binary mask
    inverted_image = binary_image.point(lambda p: 255 - p)
    
    # Save the inverted binary mask
    inverted_mask_path = os.path.join(app.config['MASK_FOLDER'], 'inverted_binary_mask.png')
    inverted_image.save(inverted_mask_path)
    
    # Copy both images to the upload folder
    shutil.copy(mask_path, app.config['UPLOAD_FOLDER'])
    shutil.copy(inverted_mask_path, app.config['UPLOAD_FOLDER'])
    
    return mask_path, inverted_mask_path

def resize_images_to_base(base_image, images):
    base_size = base_image.size
    logit(f"Base size: {base_size}")
    resized_images = [base_image]
    for img in images[1:]:
        resized_images.append(img.resize(base_size, resample=Image.Resampling.LANCZOS))
    return resized_images

@app.route('/get_images', methods=['POST','GET'])
def get_images():
    image_directory = app.config['UPLOAD_FOLDER']
    logit(f"Image directory: {image_directory}")
    image_paths = load_images(image_directory)
    image_paths_ = load_image(image_directory)

    logit(f"Loaded images: {image_paths}")
    return render_template('display_images_exp.html', image_paths=image_paths, mask_path=None, opacity=0.5, image_paths_=image_paths_)


@app.route('/play_mp3', methods=['GET', 'POST'])
def play_mp3():
    music = 'static/audio/narration.mp3'
    return render_template('play_mp3.html', music=music)
@app.route('/edit_mask', methods=['POST'])
def edit_mask():
    image_paths = request.form.getlist('image_paths')
    mask_path = request.form.get('mask_path')
    opacity = float(request.form.get('opacity', 0.5))
    return render_template('display_images_exp.html', image_paths=image_paths, mask_path=mask_path, opacity=opacity)

@app.route('/store_result', methods=['POST'])
def store_result():
    result_image_path = request.form.get('result_image')
    unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    store_path = os.path.join(app.config['STORE_FOLDER'], f'result_{unique_id}.png')
    
    # Correct the path for the result image
    result_image_path = result_image_path.replace('/static/', 'static/')
    
    # Save the result image to the store folder
    image = Image.open(result_image_path)
    image.save(store_path)
    return redirect(url_for('index'))

@app.route('/refresh-images')
def refresh_images():
    try:
        # Run the script using subprocess
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'refresh_images.py'], check=True)
        return redirect(url_for('index'))
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}'"

@app.route('/refresh-video')
def refresh_video():
    try:
        convert_images_route()    
        createvideo()
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'refresh_video.py'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'Best_FlipBook'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'diagonal_transition'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'blendem'], check=True)
        subprocess.run(['/bin/bash', 'slide'], check=True)         
        subprocess.run(['/bin/bash', 'zoomX4'], check=True)
        subprocess.run(['/bin/bash', 'zoomY4'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'vertical_scroll'], check=True)
        
        video_path = 'static/temp_exp/diagonal1.mp4'
        add_title(video_path, hex_color="#A52A2A")
        
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'joinvid'], check=True)

        return redirect(url_for('create_video'))
    
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"

@app.route('/display_resources', methods=['POST', 'GET'])
def display_resources():
    convert_images()
    # Get all image paths in the archived resources folder for jpg and png
    image_paths = glob.glob('static/archived_resources/*.jpg')
    image_paths.extend(glob.glob('static/archived_resources/*.png'))
    # List by date, last image first
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    
    # Debugging: Print the paths to verify
    logit(image_paths)
    
    return render_template('display_resources_exp.html', image_paths=image_paths)

@app.route('/copy_images', methods=['GET', 'POST'])
def copy_images():
    size_and_format_images_route()
    if request.method == 'POST':
        selected_images = request.form.getlist('selected_images')
        logit(f"Selected images: {selected_images}")
        
        # Copy the selected images to the store folder
        for image_path in selected_images:
            logit(f"Copying image: {image_path}")
            logit(f"Destination: {app.config['STORE_FOLDER']}")
            logit("----------------------")
            
            shutil.copy(image_path, app.config['STORE_FOLDER'])
        
        # Redirect to a page where you can view the stored images
        return redirect(url_for('img_processing_route'))
    
    
@app.route('/select_images', methods=['GET', 'POST'])
def select_images():
    if request.method == 'POST':
        top_image = request.form.get('top_image')
        mask_image = request.form.get('mask_image')
        bottom_image = request.form.get('bottom_image')

        if not top_image or not mask_image or not bottom_image:
            return "Please select one top image, one mask image, and one bottom image."

        # Redirect to the blend_images route with the selected images
        return redirect(url_for('blend_images', top=top_image, mask=mask_image, bottom=bottom_image))

    image_paths = get_image_paths()
    return render_template('select_images.html', image_paths=image_paths)

@app.route('/blend_images', methods=['POST', 'GET'])
def blend_images():
    # Retrieve selected images from the form
    top_image = request.form.get('top_image')
    mask_image = request.form.get('mask_image')
    bottom_image = request.form.get('bottom_image')
    opacity = float(request.form.get('opacity', 0.5))

    # Check if all required images are provided
    if not all([top_image, mask_image, bottom_image]):
        return "Please select one top image, one mask image, and one bottom image."

    # Process images
    image_paths = [top_image, mask_image, bottom_image]
    result_path = blend_images_with_grayscale_mask(image_paths, mask_image, opacity)
    mask_image ="static/masks/mask.png"
    return redirect(url_for('index'))#, image_paths=image_paths, mask_path=mask_image, opacity=opacity))
#render_template('blend_result_exp.html', result_image=result_path, image_paths=image_paths, mask_image=mask_image, opacity=opacity)

def blend_images_with_grayscale_mask(image_paths, mask_path, opacity):
    if len(image_paths) != 3:
        logit(f"Error: Expected exactly 3 image paths, got {len(image_paths)}")
        return None
    base_image_path, mask_image_path, top_image_path = image_paths
    logit(f"Base image path: {base_image_path}")
    logit(f"Mask image path: {mask_image_path}")
    logit(f"Top image path: {top_image_path}")

    base_image = Image.open(base_image_path)
    mask_image = Image.open(mask_path).convert('L')
    top_image = Image.open(top_image_path)
    base_image = base_image.resize((512,768), Image.LANCZOS)
    top_image = top_image.resize((512,768), Image.LANCZOS)
    mask_image = mask_image.resize((512,768), Image.LANCZOS)
    #base_image, top_image = resize_images_to_base(base_image, [base_image, top_image])[0], resize_images_to_base(base_image, [base_image, top_image])[1]
    blended_image = Image.composite(top_image, base_image, mask_image)

    unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    result_path = os.path.join(app.config['STORE_FOLDER'], f'result_{unique_id}.png')
    blended_image.save(result_path)
    logit(f"Blended image saved at: {result_path}")

    return result_path

@app.route('/select_mask_image', methods=['POST', 'GET'])
def select_mask_image():
    if request.method == 'POST':
        selected_image = request.form.get('selected_image')
        if not selected_image:
            return "Please select an image for masking."
        return render_template('choose_mask.html', selected_image=selected_image)
    image_paths = get_image_paths()
    return render_template('select_mask_image.html', image_paths=image_paths)
@app.route('/choose_mask', methods=['POST'])
def choose_mask():
    selected_image = request.form.get('selected_image')
    mask_type = request.form.get('mask_type')

    if not selected_image:
        return "Please select an image for masking."
    
    if mask_type == 'grayscale':
        mask_path = convert_to_grayscale(selected_image)
    elif mask_type == 'binary':
        mask_path = convert_to_binary(selected_image)
    else:
        return "Invalid mask type selected."
    #redirect to select
    return redirect(url_for('select_images'))
#render_template('select_images.html', image_paths=[selected_image], mask_path=mask_path, opacity=0.5)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/index_upload')
def index_upload():
    logit("Rendering upload form.")
    return render_template('index_upload.html')

def extract_random_frames(video_path, num_frames=25):
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        timestamps = sorted([random.uniform(0, duration) for _ in range(num_frames)])
        saved_images = []
        
        for i, timestamp in enumerate(timestamps):
            frame = video.get_frame(timestamp)
            img = Image.fromarray(frame)
            image_filename = f"frame_{i+1}.jpg"
            image_path = os.path.join(app.config['ARCHIVED_IMAGES_FOLDER'], image_filename)
            img.save(image_path)
            saved_images.append(image_filename)
        
        return saved_images
    except Exception as e:
        logit(f"Error extracting frames: {e}")
        raise

@app.route('/get_video_images', methods=['GET', 'POST'])
def get_video_images():
    if request.method == 'POST':
        url = request.form['url']
        if url:
            try:
                # Download the YouTube video
                video_path = download_youtube_video(url)
                
                # Extract 25 random frames
                images = extract_random_frames(video_path)
                
                # Redirect to display the images
                return redirect(url_for('display_images'))
            except Exception as e:
                logit(f"Error in /get_images: {e}")
                return str(e)
    
    return render_template('get_images.html')

@app.route('/images')
def display_images():
    try:
        images = os.listdir(app.config['ARCHIVED_IMAGES_FOLDER'])
        images = [os.path.join(app.config['ARCHIVED_IMAGES_FOLDER'], img) for img in images]
        return render_template('YouTube_gallery.html', images=images)
    except Exception as e:
        logit(f"Error in /images: {e}")
        return str(e)

def download_youtube_video(url):
    try:
        # Set the download options
        ydl_opts = {
            'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
            'format': 'mp4',  # Best format available
            'noplaylist': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video information and download the video
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title')
            
            # Sanitize and format the filename to remove spaces and special characters
            sanitized_title = secure_filename(video_title)
            sanitized_title = sanitized_title.replace(" ", "_")
            download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], f"{sanitized_title}.mp4")
            static_video_path = os.path.join('static', 'temp.mp4')

            # Find the downloaded file
            for root, dirs, files in os.walk(app.config['DOWNLOAD_FOLDER']):
                for file in files:
                    if file.endswith('.mp4'):
                        actual_downloaded_file = os.path.join(root, file)
                        break
                else:
                    continue
                break

            # Check if the video was downloaded correctly
            if os.path.exists(actual_downloaded_file):
                # Move the downloaded video to the static/temp.mp4 path
                shutil.move(actual_downloaded_file, static_video_path)
                logit(f"Video downloaded and moved to: {static_video_path}")
            else:
                logit(f"Downloaded file does not exist: {actual_downloaded_file}")
                raise FileNotFoundError(f"File not found: {actual_downloaded_file}")

            return static_video_path

    except Exception as e:
        logit(f"Error downloading video: {e}")
        raise
def create_feathered_image(foreground_path, output_path):
    # Load the foreground image
    foreground = cv2.imread(foreground_path)
    height, width = foreground.shape[:2]

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_foreground)

    # Create an alpha channel and a binary mask
    alpha_channel = np.zeros((height, width), dtype=np.uint8)

    if len(faces) == 0:
        print("No face detected in the image. Using the entire image with no feathering.")
        # Use the entire image with a full alpha channel
        alpha_channel = np.full((height, width), 255, dtype=np.uint8)
    else:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(alpha_channel, center, radius, 255, -1)

        # Feather the edges of the mask
        alpha_channel = cv2.GaussianBlur(alpha_channel, (101, 101), 0)

    # Add the alpha channel to the foreground image
    foreground_rgba = np.dstack((foreground, alpha_channel))

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

    # Validate dimensions
    if foreground_rgb.shape[:2] != background_rgba.shape[:2]:
        raise ValueError(f"Foreground and background dimensions do not match: {foreground_rgb.shape[:2]} vs {background_rgba.shape[:2]}")

    # Blend the images
    for i in range(3):  # For each color channel
        background_rgba[:, :, i] = (foreground_rgb[:, :, i] * alpha_channel + background_rgba[:, :, i] * (1 - alpha_channel)).astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, background_rgba)

    print(f"Composite image saved to: {output_path}")

    im = Image.open(output_path).convert('RGB')
    im.save(output_path[:-3] + 'jpg', quality=95)
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

    # Ensure the resized image is at least the target dimensions
    if resized_image.shape[0] < target_height or resized_image.shape[1] < target_width:
        resized_image = cv2.resize(image, (target_width, target_height))

    # Crop the resized image to the target dimensions
    crop_x = (resized_image.shape[1] - target_width) // 2
    crop_y = (resized_image.shape[0] - target_height) // 2
    cropped_image = resized_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

    return cropped_image

@app.route('/face_detect', methods=['POST', 'GET'])
def face_detect():
    if request.method == 'POST':
        # Check if the file part is in the request
        if 'file' not in request.files:
            flash('No file part')
            logit("Error: No file part in request.")
            return redirect(request.url)
        
        file = request.files['file']

        # Handle case where no file is selected
        if file.filename == '':
            flash('No selected file')
            logit("Error: No file selected.")
            return redirect(request.url)

        # Validate file and save if allowed
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            logit(f"File saved to {save_path}.")

            # Create a feathered PNG image from the detected face
            feathered_image_path = create_feathered_image(save_path, 'static/archived-images/feathered_face.png')
            logit(f"Feathered face image created at {feathered_image_path}.")

            # Randomly select a background image from the static/archived-images/ folder
            background_image_path = random.choice(glob.glob("static/archived-images/*.jpg"))
            logit(f"Background image selected: {background_image_path}")

            # Overlay the feathered image on the background and generate the composite image
            output_composite_path = overlay_feathered_on_background(feathered_image_path, background_image_path, 'static/archived-images/composite_image.png')
            logit(f"Composite image created at {output_composite_path}.")

            # Generate unique filenames using UUID for feathered and composite images
            feathered_image_uuid = str(uuid.uuid4()) + '.png'
            composite_image_uuid = str(uuid.uuid4()) + '.png'

            # Copy the feathered and composite images to the archived_resources directory
            feathered_image_archive_path = os.path.join('static/archived_resources', feathered_image_uuid)
            composite_image_archive_path = os.path.join('static/archived_resources', composite_image_uuid)

            shutil.copy(feathered_image_path, feathered_image_archive_path)
            shutil.copy(output_composite_path, composite_image_archive_path)
            logit(f"Feathered image archived to {feathered_image_archive_path}.")
            logit(f"Composite image archived to {composite_image_archive_path}.")

            # Return the template with paths to the generated images
            return render_template('face_detect.html', feathered_image=feathered_image_archive_path, composite_image=composite_image_archive_path)

    # Render the face_detect template on GET request or error cases
    return render_template('face_detect.html')

@app.route('/about')#, methods=['POST', 'GET'])
def about():
    return render_template('application_overview.html')
def resize_image(image_path):
    # Open the image
    image = Image.open(image_path)
    # Resize the image
    resized_image = image.resize((512, 768), Image.LANCZOS)
    # Save the resized image
    resized_image.save(image_path)
    print(f"Resized image saved at: {image_path}")
    
@app.route('/resize_all')#, methods=['POST', 'GET'])
def resize_all():
    # Resize all images in the upload folder
    image_paths = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.jpg'))
    logit(f"Image paths: {image_paths}")
    for image_path in image_paths:
        logit(f"Resizing image: {image_path}")
        resize_image(image_path)
    return redirect(url_for('index'))

import os
from datetime import datetime

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            filename = image_file.filename
            logit(f"Uploading image: {filename}")
            image_path = os.path.join('static', filename)  # Ensure only one 'static/' prefix
            image_file.save(image_path)
            return render_template('confirm_image.html', image_path=filename)  # Pass only the filename
    return render_template('upload_image.html')

@app.route('/torn_edge', methods=['POST'])
def create_torn_edge_effect():
    filename = request.form.get('image_path')
    image_path = os.path.join('static', filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"result_{timestamp}.png"
    output_path = os.path.join('static', 'archived-store', output_filename)

    logit(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")

    width, height = image.size
    logit(f"Image size: {width}x{height}")

    # Create a mask with the same size as the image
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)

    # Create a random torn edge effect for all edges
    np.random.seed(0)
    torn_edge_top = np.random.normal(0, 60, width)
    torn_edge_bottom = np.random.normal(0, 60, width)
    torn_edge_left = np.random.normal(0, 60, height)
    torn_edge_right = np.random.normal(0, 60, height)

    torn_edge_top = np.clip(torn_edge_top, -100, 100)
    torn_edge_bottom = np.clip(torn_edge_bottom, -100, 100)
    torn_edge_left = np.clip(torn_edge_left, -100, 100)
    torn_edge_right = np.clip(torn_edge_right, -100, 100)

    # Apply torn edges to the top and bottom
    for x in range(width):
        draw.line((x, 0, x, int(torn_edge_top[x])), fill=0)
        draw.line((x, height, x, height - int(torn_edge_bottom[x])), fill=0)

    # Apply torn edges to the left and right
    for y in range(height):
        draw.line((0, y, int(torn_edge_left[y]), y), fill=0)
        draw.line((width, y, width - int(torn_edge_right[y]), y), fill=0)

    # Apply Gaussian blur to smooth the edges
    logit("Applying Gaussian blur to the mask")
    mask = mask.filter(ImageFilter.GaussianBlur(2))

    logit("Applying mask to the image")
    result = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask)

    logit(f"Saving the result image: {output_path}")
    result.save(output_path, "PNG")
    #copy to upload folder
    shutil.copy(output_path, app.config['UPLOAD_FOLDER'])
    return render_template('torn_edge.html', original_image=filename, torn_image=os.path.join('archived-store', output_filename))

#!/home/jack/miniconda3/envs/cloned_base/bin/python
import io
import os
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask import send_from_directory, jsonify
import sqlite3
import datetime
from werkzeug.utils import secure_filename
import glob
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import shutil
import yt_dlp
SCAN_DATA_FILE = 'local_scan_data.json'
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
STATIC_IMAGE_DIR = 'static/local_images'
STATIC_IMAGE_DIR = 'static/local_images'
if not os.path.exists(STATIC_IMAGE_DIR):
    os.makedirs(STATIC_IMAGE_DIR)
STATIC_GALLERY_DATA_FILE = 'local_static_gallery_data.json'

app.static_folder = 'static'  # Set the static folder to 'static'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['VUPLOAD_FOLDER'] = 'static/videos'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['DATABASE'] = 'static/blog3.db'
DATABASE = app.config['DATABASE']
app.config['DATABASEF'] = 'static/functions.db'
DATABASEF = app.config['DATABASEF']

@app.route('/favicon.ico')
def favicon():
    # Logging for debug purposes
    app.logger.info("Serving favicon")
    return send_from_directory(
        directory=os.path.join(app.root_path, 'static'),
        path='favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

def logit(argvs):
    argv = argvs   
    log_file = "app_log.txt"  # Path to your log file
    timestamp = datetime.datetime.now().strftime("%A_%b-%d-%Y_%H-%M-%S")
    with open(log_file, "a") as log:
        log.write(f"{timestamp}: {argv}\n")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload_video/<int:post_id>', methods=['POST'])
def upload_video(post_id):
    if 'videoFile' not in request.files:
        logit('No file part')
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['videoFile']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['VUPLOAD_FOLDER'], filename))
        logit(f"Filename: {filename}")
        # Update the database with the filename
        update_video_filename(post_id, filename)
        flash('Video uploaded successfully')
        return redirect(url_for('post', post_id=post_id))
    else:
        flash('Allowed file types are .mp4')
        return redirect(request.url)

def update_video_filename(post_id, filename):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE post SET video_filename = ? WHERE id = ?', (filename, post_id))
        conn.commit()

# Initialize SQLite database if not exists
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS post (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE,
                content TEXT NOT NULL,
                video_filename TEXT NULL,
                image BLOB
            )
        ''')
        conn.commit()

# Function to fetch a single post by ID
@app.route('/post/<int:post_id>', methods=['GET', 'POST'])
def post(post_id):
    if request.method == 'POST':
        return upload_video(post_id)

    post = get_post(post_id)
    if not post:
        flash('Post not found')
        return redirect(url_for('home'))

    image_data = get_image_data(post_id)
    video_filename = post[4] if post[4] else None  # Adjust index based on your database schema

    return render_template('post.html', post=post, image_data=image_data, video_filename=video_filename)


def get_post(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, title, content, image, video_filename FROM post WHERE id = ? ORDER BY id DESC', (post_id,))
        post = cursor.fetchone()
    return post
# Function to fetch all posts
def get_posts(limit=None):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        if limit:
            cursor.execute('SELECT id, title, content, image, video_filename FROM post ORDER BY id DESC LIMIT ?', (limit,))
        else:
            cursor.execute('SELECT id, title, content, image, video_filename FROM post ORDER BY id DESC')
        posts = cursor.fetchall()
    return posts
def get_intro(limit=1):
    logit(f"Fetching post with limit={limit}")
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        try:
            if limit:
                logit("Fetching post with id = 864")
                cursor.execute('SELECT id, title, content, image, video_filename FROM post WHERE id = ?', (864,))
            else:
                logit("Fetching all posts ordered by id DESC")
                cursor.execute('SELECT id, title, content, image, video_filename FROM post ORDER BY id DESC')
                
            posts = cursor.fetchall()
            logit(f"Fetched posts: {posts}")
            return posts
        except sqlite3.OperationalError as e:
            logit(f"SQLite error occurred: {e}")
            raise
# Function to fetch image data
def get_image(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT image FROM post WHERE id = ?', (post_id,))
        post = cursor.fetchone()
        if post and post[0]:
            return post[0]  # Return base64 encoded image data
        return None

@app.route('/home2')
def home2():
    posts = get_posts(limit=6) 
    for post in posts:
        logit(post[3])# Limit to last 6 posts
    return render_template('home.html', posts=posts)

@app.route('/new', methods=['GET', 'POST'])
def new_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        image = request.files['image'].read() if 'image' in request.files and request.files['image'].filename != '' else None
        if image:
            image = base64.b64encode(image).decode('utf-8')
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO post (title, content, image) VALUES (?, ?, ?)', (title, content, image))
            conn.commit()
        flash('Post created successfully!', 'success')
        return redirect(url_for('home2'))
    return render_template('new_post.html')

@app.route('/edit/<int:post_id>', methods=['GET', 'POST'])
def edit_post(post_id):
    post = get_post(post_id)
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        image_data = get_image(post_id)  # Get the current image data
        if 'image' in request.files and request.files['image'].filename != '':
            image = request.files['image'].read()
            image_data = base64.b64encode(image).decode('utf-8')  # Update with new image data
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE post SET title = ?, content = ?, image = ? WHERE id = ?', (title, content, image_data, post_id))
            conn.commit()
        flash('Post updated successfully!', 'success')
        return redirect(url_for('post', post_id=post_id))
    return render_template('edit_post.html', post=post)

@app.route('/contents')
def contents():
    posts = get_posts()
    contents_data = []
    for post in posts:
        excerpt = post[2][:300] + '...' if len(post[2]) > 300 else post[2]  # Assuming content is in the third column (index 2)
        contents_data.append({
            'id': post[0],
            'title': post[1],
            'excerpt': excerpt
        })
    return render_template('contents.html', contents_data=contents_data)

@app.route('/delete/<int:post_id>', methods=['POST'])
def delete_post(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM post WHERE id = ?', (post_id,))
        conn.commit()
    flash('Post deleted successfully!', 'success')
    return redirect(url_for('home'))

def load_txt_files(directory):
    init_db()  # Initialize the SQLite database if not already created
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    title = os.path.splitext(filename)[0]
                    content = file.read()
                    cursor.execute('SELECT id FROM post WHERE title = ? ORDER BY id DESC', (title,))
                    existing_post = cursor.fetchone()
                    if not existing_post:
                        cursor.execute('INSERT INTO post (title, content) VALUES (?, ?)', (title, content))
                        conn.commit()
                        print(f'Added post: {title}')
                    else:
                        print(f'Skipped existing post: {title}')
    except sqlite3.Error as e:
        print(f'SQLite error: {e}')
    finally:
        conn.close()

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_terms = request.form['search_terms']
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Define the search terms
        search_terms = search_terms.split(",")  # Split by comma to get individual search terms
        results = []
        
        # Construct the WHERE clause for the SQL query to filter rows based on all search terms
        where_conditions = []
        for term in search_terms:
            where_conditions.append(f"content LIKE ?")
        
        where_clause = " AND ".join(where_conditions)
        
        # Create a tuple of search terms with wildcard characters for the SQL query
        search_terms_tuple = tuple(f"%{term.strip()}%" for term in search_terms)
        
        # Execute the SELECT query with the constructed WHERE clause
        query = f"SELECT ROWID, title, content, image, video_filename FROM post WHERE {where_clause} ORDER BY ROWID DESC"
        rows = cursor.execute(query, search_terms_tuple)

        
        for row in rows:
            results.append((row[0], row[1], row[2], row[3], row[4]))
        
        conn.close()
        return render_template('search.html', results=results)
    
    return render_template('search.html', results=[])


def get_image_data(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT image FROM post WHERE id = ?', (post_id,))
        post = cursor.fetchone()
        if post and post[0]:
            return base64.b64decode(post[0])  # Decode base64 to binary
        else:
            return None

@app.route('/post/<int:post_id>', methods=['GET', 'POST'])
def show_post(post_id):
    if request.method == 'POST':
        if 'videoFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['videoFile']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['VUPLOAD_FOLDER'], filename))
            
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE post SET video_filename = ? WHERE id = ?', (filename, post_id))
                conn.commit()
                flash('Video uploaded successfully')

            return redirect(url_for('show_post', post_id=post_id))

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, title, content, image, video_filename FROM post WHERE id = ? ORDER BY id DESC', (post_id,))
        post = cursor.fetchone()
        if not post:
            flash('Post not found')
            return redirect(url_for('home'))
        
        image_data = base64.b64decode(post[3]) if post[3] else None
        video_filename = post[4] if post[4] else None
    logit(f"video_filename: {video_filename}")
    return render_template('post.html', post=post, image_data=image_data, video_filename=video_filename)

@app.route('/image/<int:post_id>')
def view_image(post_id):
    image_data = get_image_data(post_id)
    if image_data:
        return send_file(io.BytesIO(image_data), mimetype='image/jpeg')
    else:
        return "No image found", 404

TEXT_FILES_DIR = "static/TEXT" 
# Index route to display existing text files and create new ones
@app.route("/edit_text", methods=["GET", "POST"])
def edit_text():

    if request.method == "POST":
        filename = request.form["filename"]
        text = request.form["text"]
        save_text_to_file(filename, text)
        return redirect(url_for("edit_text"))
    else:
        # Path to the file containing list of file paths
        text_files = os.listdir(TEXT_FILES_DIR)
        text_directory='static/TEXT'
        files = sorted(text_files, key=lambda x: os.path.getmtime(os.path.join(text_directory, x)), reverse=True)
        #files=glob.glob('static/TEXT/*.txt')
        logit(f'files 1: {files}')  
        # Call the function to list files by creation time
        #files = list_files_by_creation_time(files)
        logit(f'files 2: {files}')
        return render_template("edit_text.html", files=files)
 # Route to edit a text file
@app.route("/edit/<filename>", methods=["GET", "POST"])
def edit(filename):
    if request.method == "POST":
        text = request.form["text"]
        save_text_to_file(filename, text)
        return redirect(url_for("index"))
    else:
        text = read_text_from_file(filename)
        return render_template("edit.html", filename=filename, text=text)
# Route to delete a text file
@app.route("/delete/<filename>")
def delete(filename):
    filepath = os.path.join(TEXT_FILES_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        logit(f"File deleted: {filename}")
    return redirect(url_for("index"))


def list_files_by_creation_time(file_paths):
    """
    List files by their creation time, oldest first.
    
    Args:
    file_paths (list): List of file paths.
    
    Returns:
    list: List of file paths sorted by creation time.
    """
    # Log the start of the function
    logit('Listing files by creation time...')
    
    # Create a dictionary to store file paths and their creation times
    file_creation_times = {}
    
    # Iterate through each file path
    for file_path in file_paths:
        # Get the creation time of the file
        try:
            creation_time = os.path.getctime(file_path)
            # Store the file path and its creation time in the dictionary
            file_creation_times[file_path] = creation_time
        except FileNotFoundError:
            # Log a warning if the file is not found
            logit(f'File not found: {file_path}')
    
    # Sort the dictionary by creation time
    sorted_files = sorted(file_creation_times.items(), key=lambda x: x[1],reverse=True)
    
    # Extract the file paths from the sorted list
    sorted_file_paths = [file_path for file_path, _ in sorted_files]
    
    # Log the end of the function
    logit('File listing complete.')
    
    # Return the sorted file paths
    return sorted_file_paths
def read_text_from_file(filename):
    filepath = os.path.join(TEXT_FILES_DIR, filename)
    with open(filepath, "r") as file:
        text = file.read()
        logit(f"Text read from file: {filename}")
        return text
    

@app.route('/generate', methods=['POST'])
def generate_text():
    input_text = request.form['input_text']
    generated_text = generate_text_with_model(input_text)
    logit(f"Generated text: {generated_text}")
    return jsonify({'generated_text': generated_text})

def generate_text_with_model(input_text):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    sample_output = model.generate(
        input_ids, 
        max_length=500, 
        temperature=0.8, 
        top_p=0.9, 
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return generated_text

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    return html_content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 Text Generation</title>
    <link rel="stylesheet" href="static/dark.css">
    <style>
        textarea {
            width: 100% !important;
            height: 60px !important;
            margin: 10px 0;
            padding: 10px;
            font-size: 16px;
        }
        input[type="submit"] {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        pre {
            background-color: darkgray;
            padding: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 22px;
        }
        #text {
            background-color: black;
            margin-top: 20px;
            font-size: 24px;
        }
    
        </style>
</head>
<body>
    <h1>GPT-2 Text Generation</h1>
    <!-- Add link home -->
    <a href="/">Home</a>
    <form id="inputForm">
        <label for="input_text">Enter Input Text:</label><br>
        <textarea class="small" id="input_text" name="input_text"></textarea><br>
        <input type="submit" value="Generate Text">
    </form>
    <pre style="color:black;" id="generated_text"></pre>
    <script>
        document.getElementById('inputForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            const formData = new FormData(this);
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            document.getElementById('generated_text').innerHTML = '<h2>Generated Text:</h2>' + data.generated_text;
        });
    </script>
</body>
</html>
"""
def save_static_gallery_data(data):
    with open(STATIC_GALLERY_DATA_FILE, 'w') as f:
        json.dump(data, f)
    app.logger.info(f'Static gallery data saved to {STATIC_GALLERY_DATA_FILE}')

def load_static_gallery_data():
    # Scan directories if static gallery data file is missing or empty
    if not os.path.exists(STATIC_GALLERY_DATA_FILE):
        app.logger.info(f'{STATIC_GALLERY_DATA_FILE} not found. Scanning directories.')
        scan_directories() 
    else:           
        with open(STATIC_GALLERY_DATA_FILE, 'r') as f:
            data = json.load(f)
            app.logger.info(f'Static gallery data loaded from {STATIC_GALLERY_DATA_FILE}')
            return data
    return None

def scan_directories():
    image_dirs = []
    for root, dirs, files in os.walk('static/'):  # You can specify a specific directory to start scanning
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        if len(image_files) >= 20:
            image_dirs.append({
                'path': root,
                'images': image_files
            })
            app.logger.info(f'Found {len(image_files)} images in directory: {root}')
    return image_dirs

def save_scan_data(data):
    with open(SCAN_DATA_FILE, 'w') as f:
        json.dump(data, f)
    app.logger.info(f'Scan data saved to {SCAN_DATA_FILE}')

def load_scan_data():
    if os.path.exists(SCAN_DATA_FILE):
        with open(SCAN_DATA_FILE, 'r') as f:
            data = json.load(f)
            app.logger.info(f'Scan data loaded from {SCAN_DATA_FILE}')
            return data
    app.logger.info(f'{SCAN_DATA_FILE} not found.')
    return None

def select_random_images(image_dirs):
    gallery_data = []
    for dir_data in image_dirs:
        images = dir_data['images']
        if len(images) >= 20:
            sample_images = random.sample(images, 3)
            gallery_data.append({
                'directory': dir_data['path'],
                'images': [os.path.join(dir_data['path'], img) for img in sample_images]
            })
            app.logger.info(f'Selected images from directory: {dir_data["path"]}')
    return gallery_data

def copy_images_to_static(gallery_data):
    if not os.path.exists(STATIC_IMAGE_DIR):
        os.makedirs(STATIC_IMAGE_DIR)
        app.logger.info(f'Created directory: {STATIC_IMAGE_DIR}')

    static_image_paths = []
    for item in gallery_data:
        static_images = []
        for img_path in item['images']:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(STATIC_IMAGE_DIR, img_name)
            shutil.copy(img_path, dest_path)
            static_images.append(dest_path)
            app.logger.info(f'Copied image {img_name} to {STATIC_IMAGE_DIR}')
        static_image_paths.append({
            'directory': item['directory'],
            'images': static_images
        })
    return static_image_paths

@app.route('/gallery')
def gallery():
    scan_data = load_scan_data()
    if not scan_data:
        app.logger.info('No scan data found. Scanning directories.')
        scan_data = scan_directories()
        save_scan_data(scan_data)
        gallery_data = select_random_images(scan_data)
        static_gallery_data = copy_images_to_static(gallery_data)
    else:
        static_gallery_data = load_static_gallery_data()
        if not static_gallery_data:
            app.logger.info('No static gallery data found. Creating new static gallery data.')
            gallery_data = select_random_images(scan_data)
            static_gallery_data = copy_images_to_static(gallery_data)
            save_static_gallery_data(static_gallery_data)
    return render_template('gallery.html', gallery_data=static_gallery_data)


@app.route('/remove_images', methods=['GET', 'POST'])
def remove_images():
    folder = 'static/archived-store/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            logit(f"Removed file: {file_path}")
    return redirect(url_for('index'))

# Directory containing the images
archived_images_dir = 'static/archived-store'
#archived_images_dir = 'static/archived-images/'  # Update this path as needed


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        logit("No file part in request.")
        return redirect(request.url)

    file = request.files['file']

    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        logit("No file selected.")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        logit(f"File saved to {save_path}.")
        return redirect(url_for('uploaded_file', filename=filename))
    else:
        flash('File type not allowed')
        logit("File type not allowed.")
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # You can create a route to handle what happens after a file is uploaded successfully
    logit(f"File uploaded: {filename}.")
    return render_template ('uploaded_image.html', filename=filename)

    #return f"File {filename} uploaded successfully!"
    #-------------------------------
@app.route('/remove_image', methods=['GET', 'POST'])
def remove_image():
    folder = 'static/archived-images/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            logit(f"Removed file: {file_path}")
    return redirect(url_for('index'))

# Directory containing the images
archived_images = 'static/archived-images'
@app.route('/clean_archives', methods=['GET', 'POST'])
def clean_archives():
    if request.method == 'POST':
        # Get selected images
        selected_images = request.form.getlist('selected_images')
        #list of images sorted by creation time

        
        # Remove selected images
        for image in selected_images:
            image_path = os.path.join(archived_images, image)
            if os.path.exists(image_path):
                os.remove(image_path)
                logit(f"Removed image: {image_path}")
        
        return redirect(url_for('clean_archives'))
    
    # Get list of images in the directory (png and jpg)
    images = [os.path.basename(img) for img in glob.glob(os.path.join(archived_images, '*.png'))]
    images += [os.path.basename(img) for img in glob.glob(os.path.join(archived_images, '*.jpg'))]
    images = sorted(images, key=lambda x: os.path.getmtime(os.path.join(archived_images, x)), reverse=True)
    logit(f"clean_archives_Images: {images}")
    return render_template('clean_archives.html', images=images)

def create_backup_folder():
    backup_folder = os.path.join(os.getcwd(), "static", "Backups")
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
        logit(f"Backup folder created at: {backup_folder}")
@app.route('/edit_files', methods=['GET', 'POST'])
def edit_files():
    filename = request.args.get('filename', '')
    directory_path = "."
    PATH = os.getcwd()
    print(f"Current Directory:, {PATH}")
    full_path = os.path.join(PATH, directory_path, filename)

    if not os.path.exists(filename):
        return "File not found", 404

    if request.method == 'POST':
        content = request.form.get('content')
        
        if content is not None:
            date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            new_filename = f"{os.path.splitext(filename)[0]}_{date_str}.txt"
            logit(f'File edited and saved as: {new_filename}')

            with open(os.path.join(directory_path, new_filename), 'w') as new_file:
                new_file.write(content)

            logit(f'File edited and saved as: {new_filename}')

            return send_file(os.path.join(directory_path, new_filename), as_attachment=True)

    with open(full_path, 'r') as file:
        content = file.read()

    return render_template('edit_files.html', filename=filename, content=content)

@app.route('/edit_html', methods=['POST', 'GET'])
def edit_html():
    path = currentDir+"/templates/"
    if request.method == 'POST':
        if 'load_file' in request.form:
            selected_file = request.form['selected_file']
            logit(f"Received request to load file: {selected_file}")
            file_path = os.path.join(path, selected_file)
            logit(f"Received request to load file: {file_path}")

            # Load the HTML content from the file
            html_content = load_html_file(file_path)
            logit(f"Loaded HTML content: {html_content}")

            # Pass the loaded content to the template
            return render_template('choose_file.html', files=choose_html(), file_path=file_path, html_content=html_content)

        elif 'edited_content' in request.form:
            edited_content = request.form['edited_content']
            file_path = os.path.join(path, request.form['selected_file'])

            # Save the edited HTML content
            with open(file_path, "w") as file:
                logit(f"Saving edited content to file: {file_path}")
                logit(f"Edited content: {edited_content}")
                file.write(edited_content)

    return render_template('choose_file.html', files=choose_html())


def load_html_file(file_path):
    with open(file_path, "r") as file:
        html_content = file.read()
        logit(f"Loaded HTML content from file: {file_path}")
        logit(f"HTML content: {html_content}")
    return html_content

def edit_html_file(file_path, new_content):
    with open(file_path, "w") as file:
        file.write(new_content)

def save_html_file(file_path, soup):
    with open(file_path, "w") as file:
        file.write(str(soup))
def choose_html():
    path = currentDir+"/templates/"
    files = glob.glob(path + "*.html")  # get all files in the directory

    if not files:
        print("No files found")
        return None
    return files

#remove just images from the static folder static/archived-store/ leave subfolders
TEMPLATE_DIR = 'templates'
@app.route('/html_index')
def html_index():
    # List all HTML files in the templates directory
    template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.html')]
    return render_template('html_index.html', template_files=template_files)

@app.route('/edit/<filename>', methods=['GET', 'POST'])
def edit_template(filename):
    file_path = os.path.join(TEMPLATE_DIR, filename)
    if request.method == 'POST':
        # Get the edited content from the form
        content = request.form['content']
        try:
            with open(file_path, 'w') as file:
                file.write(content)
            flash('Template updated successfully!', 'success')
        except Exception as e:
            flash(f'Error updating template: {str(e)}', 'danger')
        return redirect(url_for('index'))

    try:
        # Read the content of the template file
        with open(file_path, 'r') as file:
            content = file.read()
    except Exception as e:
        flash(f'Error reading template: {str(e)}', 'danger')
        return redirect(url_for('index'))
    
    return render_template('edit_html.html', filename=filename, content=content)

def feather_image(image, radius=50):
    """Applies a feathered transparency effect to the left and right edges of an image."""
    logit(f"Applying feather effect with radius {radius} to image of size {image.size}")
    
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (radius, 0, image.width - radius, image.height))
    mask = mask.filter(ImageFilter.GaussianBlur(radius))
    
    image.putalpha(mask)
    return image

def create_seamless_image(images, feather_radius=5, overlap=100):
    """Creates a seamless image by blending the provided images with feathered edges and overlap."""
    total_width = sum(img.width for img in images) - overlap * (len(images) - 1)
    max_height = max(img.height for img in images)

    logit(f"Creating combined image of size {total_width}x{max_height}")
    
    combined_image = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for i, img in enumerate(images):
        feathered_img = feather_image(img, feather_radius)
        combined_image.paste(feathered_img, (x_offset, 0), feathered_img)
        x_offset += img.width - overlap
        logit(f"Image {i+1} pasted at position {x_offset}")

    return combined_image

def make_scrolling_video(image_path, output_video_path, video_duration=10, video_size=(512, 768)):
    """Creates a video by scrolling across the image from left to right."""
    
    logit(f"Loading image from {image_path}")
    
    image = ImageClip(image_path)

    def scroll_func(get_frame, t):
        x = int((image.size[0] - video_size[0]) * t / video_duration)
        return get_frame(t)[0:video_size[1], x:x+video_size[0]]
    
    video = VideoClip(lambda t: scroll_func(image.get_frame, t), duration=video_duration)
    video = video.set_fps(24)

    logit(f"Saving video to {output_video_path}")
    video.write_videofile(output_video_path, codec='libx264', audio=False)

@app.route('/create_video', methods=['POST', 'GET'])
def create_video():
    """Endpoint to create a scrolling video from a set of images."""
    #copy static/temp_exp/TEMP2.mp4 static/temp_exp/diagonal1.mp4
    #shutil.copy("static/temp_exp/TEMP2.mp4", "static/temp_exp/diagonal1.mp4")
    try:
        vid_directory = 'static/archived-store'

        # Get all image files in the directory
        #image_files = glob.glob(os.path.join(vid_directory, '*.png'))
        image_files = glob.glob(os.path.join(vid_directory, '*.png')) + glob.glob(os.path.join(vid_directory, '*.jpg'))

        if not image_files:
            logit("No image files found.")
            return jsonify({"error": "No image files found."}), 404

        # Sort files by modification time
        image_files = sorted(image_files, key=os.path.getmtime, reverse=True)

        if len(image_files) < 6:
            logit("Less than 6 images found. Adjusting the number of selected images.")

        images = [Image.open(img).convert('RGBA').resize((512, 768), resample=Image.LANCZOS) for img in image_files]

        # Create the seamless image
        seamless_image_path = 'static/seamless_image.png'
        seamless_image = create_seamless_image(images, feather_radius=10, overlap=100)
        seamless_image.save(seamless_image_path)
        logit(f"Seamless image saved as {seamless_image_path}")

        # Create the scrolling video
        video_path = 'static/seamless_video.mp4'
        video_duration = 34
        video_size = (512, 768)
        make_scrolling_video(seamless_image_path, video_path, video_duration, video_size)
        
        # Optionally run external process if needed
        add_title_image(video_path, hex_color="#A52A2A")
        return redirect(url_for('index'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
def createvideo():
    try:
        vid_directory = 'static/archived-store'
        # Get all image files in the directory
        #image_files = glob.glob(os.path.join(vid_directory, '*.png'))
        image_files = glob.glob(os.path.join(vid_directory, '*.png')) + glob.glob(os.path.join(vid_directory, '*.jpg'))

        if not image_files:
            logit("No image files found.")
            return jsonify({"error": "No image files found."}), 404

        # Sort files by modification time
        image_files = sorted(image_files, key=os.path.getmtime, reverse=True)

        if len(image_files) < 6:
            logit("Less than 6 images found. Adjusting the number of selected images.")

        images = [Image.open(img).convert('RGBA').resize((512, 768), resample=Image.LANCZOS) for img in image_files[:8]]

        # Create the seamless image
        seamless_image_path = 'static/seamless_image.png'
        seamless_image = create_seamless_image(images, feather_radius=10, overlap=100)
        seamless_image.save(seamless_image_path)
        logit(f"Seamless image saved as {seamless_image_path}")

        # Create the scrolling video
        video_path = 'static/seamless_videoX.mp4'
        video_duration = 34
        video_size = (512, 768)
        make_scrolling_video(seamless_image_path, video_path, video_duration, video_size)
        
        # Optionally run external process if needed
        add_title_image(video_path, hex_color="#A52A2A")
        return redirect(url_for('index'))
    except Exception as e:
        logit(f"Error creating video: {e}")
        return jsonify({"error": str(e)}), 500


def add_title_image(video_path, hex_color = "#A52A2A"):
    hex_color=random.choice(["#A52A2A","#ad1f1f","#16765c","#7a4111","#9b1050","#8e215d","#2656ca"])
    # Define the directory path
    directory_path = "temp"
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If not, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 
    # Load the video file and title image
    video_clip = VideoFileClip(video_path)
    print(video_clip.size)
    # how do i get the width and height of the video
    width, height = video_clip.size
    get_duration = video_clip.duration
    print(get_duration, width, height)
    title_image_path = "static/assets/512x568_border.png"
    # Set the desired size of the padded video (e.g., video width + padding, video height + padding)
    padded_size = (width + 50, height + 50)

    # Calculate the position for centering the video within the larger frame
    x_position = (padded_size[0] - video_clip.size[0]) / 2
    y_position = (padded_size[1] - video_clip.size[1]) / 2
    #hex_color = "#09723c"
    # Remove the '#' and split the hex code into R, G, and B components
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Create an RGB tuple
    rgb_tuple = (r, g, b)

    # Create a blue ColorClip as the background
    blue_background = ColorClip(padded_size, color=rgb_tuple)

    # Add the video clip on top of the red background
    padded_video_clip = CompositeVideoClip([blue_background, video_clip.set_position((x_position, y_position))])
    padded_video_clip = padded_video_clip.set_duration(video_clip.duration)
    #title_image_path = "/home/jack/Desktop/EXPER/static/assets/Title_Image02.png"
    # Load the title image
    title_image = ImageClip(title_image_path)

    # Set the duration of the title image
    title_duration = video_clip.duration
    title_image = title_image.set_duration(title_duration)

    print(video_clip.size)
    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("left", "top"))
    title_image = title_image.set_position((0, -5))
    #video_clip.size = (620,620)
    title_image = title_image.resize(padded_video_clip.size)

    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("center", "center")).resize(video_clip.size)

    # Create a composite video clip with the title image overlay
    composite_clip = CompositeVideoClip([padded_video_clip, title_image])
    # Limit the length to video duration
    composite_clip = composite_clip.set_duration(video_clip.duration)
    # Load a random background music
    mp3_files = glob.glob("/mnt/HDD500/collections/music_dark/*.mp3")
    random.shuffle(mp3_files)

    # Now choose a random MP3 file from the shuffled list
    mp_music = random.choice(mp3_files)
    get_duration = AudioFileClip(mp_music).duration
    # Load the background music without setting duration
    music_clip = AudioFileClip(mp_music)
    # Fade in and out the background music
    #music duration is same as video
    music_clip = music_clip.set_duration(video_clip.duration)
    # Fade in and out the background music
    fade_duration = 1.0
    music_clip = music_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)
    # Set the audio of the composite clip to the background music
    composite_clip = composite_clip.set_audio(music_clip)
    uid = uuid.uuid4().hex
    output_path = 'static/temp_exp/TEMP2X.mp4'
    # Export the final video with the background music
    composite_clip.write_videofile(output_path)
    mp4_file =  f"/mnt/HDD500/collections/vids/Ready_Post_{uid}.mp4"
    shutil.copyfile(output_path, mp4_file)     
    print(mp4_file)
    VIDEO = output_path
    return VIDEO
def add_title(video_path, hex_color = "#A52A2A"):
    hex_color=random.choice(["#A52A2A","#ad1f1f","#16765c","#7a4111","#9b1050","#8e215d","#2656ca"])
    # Define the directory path
    directory_path = "tempp"
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If not, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 
    # Load the video file and title image
    video_clip = VideoFileClip(video_path)
    print(video_clip.size)
    # how do i get the width and height of the video
    width, height = video_clip.size
    get_duration = video_clip.duration
    print(get_duration, width, height)
    title_image_path = "static/assets/512x568_border.png"
    # Set the desired size of the padded video (e.g., video width + padding, video height + padding)
    padded_size = (width + 50, height + 50)

    # Calculate the position for centering the video within the larger frame
    x_position = (padded_size[0] - video_clip.size[0]) / 2
    y_position = (padded_size[1] - video_clip.size[1]) / 2
    #hex_color = "#09723c"
    # Remove the '#' and split the hex code into R, G, and B components
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Create an RGB tuple
    rgb_tuple = (r, g, b)

    # Create a blue ColorClip as the background
    blue_background = ColorClip(padded_size, color=rgb_tuple)

    # Add the video clip on top of the red background
    padded_video_clip = CompositeVideoClip([blue_background, video_clip.set_position((x_position, y_position))])
    padded_video_clip = padded_video_clip.set_duration(video_clip.duration)
    #title_image_path = "/home/jack/Desktop/EXPER/static/assets/Title_Image02.png"
    # Load the title image
    title_image = ImageClip(title_image_path)

    # Set the duration of the title image
    title_duration = video_clip.duration
    title_image = title_image.set_duration(title_duration)

    print(video_clip.size)
    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("left", "top"))
    title_image = title_image.set_position((0, -5))
    #video_clip.size = (620,620)
    title_image = title_image.resize(padded_video_clip.size)

    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("center", "center")).resize(video_clip.size)

    # Create a composite video clip with the title image overlay
    composite_clip = CompositeVideoClip([padded_video_clip, title_image])
    # Limit the length to video duration
    composite_clip = composite_clip.set_duration(video_clip.duration)
    # Load a random background music
    mp3_files = glob.glob("/mnt/HDD500/collections/music_dark/*.mp3")
    random.shuffle(mp3_files)

    # Now choose a random MP3 file from the shuffled list
    mp_music = random.choice(mp3_files)
    get_duration = AudioFileClip(mp_music).duration
    # Load the background music without setting duration
    music_clip = AudioFileClip(mp_music)
    # Fade in and out the background music
    #music duration is same as video
    music_clip = music_clip.set_duration(video_clip.duration)
    # Fade in and out the background music
    fade_duration = 1.0
    music_clip = music_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)
    # Set the audio of the composite clip to the background music
    composite_clip = composite_clip.set_audio(music_clip)
    uid = uuid.uuid4().hex
    output_path = 'static/temp_exp/TEMP1X.mp4'
    # Export the final video with the background music
    composite_clip.write_videofile(output_path)
    mp4_file =  f"/mnt/HDD500/collections/vids/Ready_Post_{uid}.mp4"
    shutil.copyfile(output_path, mp4_file)     
    print(mp4_file)
    VIDEO = output_path
    return VIDEO
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def next_points(point, imgsize, avoid_points=[], shuffle=True):
    point_list = [p for p in 
                  [(point[0], point[1]+1), (point[0], point[1]-1), 
                   (point[0]+1, point[1]), (point[0]-1, point[1])]
                  if 0 < p[0] < imgsize[0]//2 and 0 < p[1] < imgsize[1] and p not in avoid_points]

    if shuffle:
        random.shuffle(point_list)

    return point_list

def degrade_color(color, degradation=10):
    return tuple(min(c + degradation, 255) for c in color)

def spread(img, point, color, max_white=100, degradation=10):
    if color[0] <= max_white and img.getpixel(point)[0] > color[0]:
        img.putpixel(point, color)
        points = next_points(point, img.size, shuffle=False)
        color = degrade_color(color, degradation)
        for point in points:
            spread(img, point, color)

def binarize_array(numpy_array, threshold=200):
    return np.where(numpy_array > threshold, 255, 0)

def process_image(seed_count, seed_max_size, imgsize=(510, 766), count=0):
    margin_h, margin_v = 60, 60
    color = (0, 0, 0)
    img = Image.new("RGB", imgsize, "white")
    old_points = []
    posible_root_points = []

    for seed in range(seed_count):
        point = None
        while not point or point in old_points:
            point = (random.randrange(0 + margin_h, imgsize[0]//2), 
                     random.randrange(0 + margin_v, imgsize[1] - margin_v))
        old_points.append(point)
        posible_root_points.append(point)
        img.putpixel(point, color)

        seedsize = random.randrange(0, seed_max_size)
        flow = 0
        for progress in range(seedsize):
            flow += 1
            points = next_points(point, imgsize, old_points)
            try:
                point = points.pop()
            except IndexError:
                posible_root_points.remove(point)
                for idx in reversed(range(len(posible_root_points))):
                    points = next_points(posible_root_points[idx], imgsize, old_points)
                    try:
                        point = points.pop()
                        flow = 0
                        break
                    except IndexError:
                        posible_root_points.pop()
                if not point:
                    break

            old_points.append(point)
            posible_root_points.append(point)
            img.putpixel(point, color)

            for surr_point in points:
                spread(img, surr_point, degrade_color(color))

    cropped = img.crop((0, 0, imgsize[0]//2, imgsize[1]))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.paste(cropped, (0, 0, imgsize[0]//2, imgsize[1]))
    img = img.filter(ImageFilter.GaussianBlur(radius=10))
    
    filename0 = "static/images/blot.png"
    img.save(filename0)

    im_grey = img.convert('LA')
    mean = np.mean(np.array(im_grey)[:, :, 0])

    image_file = Image.open(filename0)
    imagex = image_file.convert('L')
    imagey = np.array(imagex)
    imagez = binarize_array(imagey, mean)

    temp_filename = "static/images/tmmmp.png"
    cv2.imwrite(temp_filename, imagez)

    final_filename = time.strftime("static/archived-images/GOODblots%Y%m%d%H%M%S.png")
    ImageOps.expand(Image.open(temp_filename).convert("L"), border=1, fill='white').save(final_filename)

    print("GoodBlot: ", count)
    return final_filename

@app.route('/inkblot')
def rorschach():
    ensure_dir_exists("static/images")
    ensure_dir_exists("static/blot")

    # Generate the inkblots
    inkblot_images = []
    for count in range(2):  # Generate 2 inkblots as an example
        seed_count = random.randint(6, 10)
        seed_max_size = random.randint(5000, 16000)
        inkblot_image = process_image(seed_count, seed_max_size, count=count)
        inkblot_images.append(inkblot_image)

    # Pass the image paths to the template
    return render_template('Rorschach.html', inkblot_images=inkblot_images)

# Function for image processing
def process_image(image_path):
    img = imread(image_path)
    labels = segmentation.slic(img, compactness=30, n_segments=400)
    g = future.graph.rag_mean_color(img, labels)

    def weight_boundary(graph, src, dst, n):
        default = {'weight': 0.0, 'count': 0}
        count_src = graph[src].get(n, default)['count']
        count_dst = graph[dst].get(n, default)['count']
        weight_src = graph[src].get(n, default)['weight']
        weight_dst = graph[dst].get(n, default)['weight']
        count = count_src + count_dst
        return {
            'count': count,
            'weight': (count_src * weight_src + count_dst * weight_dst) / count
        }

    def merge_boundary(graph, src, dst):
        pass

    labels2 = future.graph.merge_hierarchical(labels, g, thresh=0.08, rag_copy=False,
                                              in_place_merge=True,
                                              merge_func=merge_boundary,
                                              weight_func=weight_boundary)

    out = color.label2rgb(labels2, img, kind='avg')
    
    # Save the processed image
    output_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{time.time()}.png")
    imsave(output_filename, out)
    return output_filename

# Route for the main page
@app.route('/uploadfile', methods=['GET', 'POST'])
def uploadfile():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image and get the output path
            processed_image_path = process_image(filepath)
            
            return redirect(url_for('uploaded_file', filename=os.path.basename(processed_image_path)))
    return '''
    <!doctype html>
    <title>Upload an image</title>
    <h1>Upload an image for segmentation processing</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
    
try:
    os.makedirs("static/outlines")
except FileExistsError:
    # directory already exists
    pass
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

#image = cv2.imread('mahotastest/orig-color.png')
def change_extension(orig_file,new_extension):
    p = change_ext(orig_file)
    new_name = p.rename(p.with_suffix(new_extension))
    return new_name
    
def FilenameByTime(directory):
    timestamp = str(time.time()).replace('.', '')
    filename = f"{directory}/{timestamp}.png"
    return filename

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

import os
import cv2
from PIL import Image
import shutil

def outlineJ(filename, sigma=0.33):
    # Load the image and apply Canny edge detection
    image = cv2.imread(filename)
    edged = auto_canny(image, sigma=sigma)
    
    # Invert the colors for black-and-white outlines
    inverted = cv2.bitwise_not(edged)
    
    # Paths to save the images
    temp_path = "static/outlines/temp2.png"
    outline_path = "static/outlines/outlined.png"
    transparent_path = "static/outlines/transparent_outline.png"

    # Save the inverted black-and-white image
    cv2.imwrite(temp_path, inverted)

    # Open the black-and-white outline image
    frontimage = Image.open(temp_path).convert("RGBA")  # Load as RGBA for transparency

    # Process to create the black outline with transparent background
    datas = frontimage.getdata()

    newData = []
    for item in datas:
        # If the pixel is white, make it transparent
        if item[0] > 200 and item[1] > 200 and item[2] > 200:  # Adjust as necessary
            newData.append((255, 255, 255, 0))
        else:
            newData.append((0, 0, 0, 255))  # Keep black as is

    frontimage.putdata(newData)
    frontimage.save(transparent_path)  # Save the transparent outline

    # Open the original image to apply the outline
    background = Image.open(filename).convert("RGBA")
    background.paste(frontimage, (3, 3), frontimage)  # Paste with transparency
    
    # Save the outlined image
    background.save(outline_path)

    # Save the outline on a white background
    outline_on_white = Image.new("RGBA", background.size, "WHITE")
    outline_on_white.paste(frontimage, (0, 0), frontimage)
    outline_on_white.save(temp_path)

    # Save the images with timestamps
    unique_id = uuid.uuid4().hex
    #savefile = f"static/outlines/{unique_id}.png"
    
    savefile = FilenameByTime("static/outlines")
    shutil.copyfile(transparent_path, f"static/archived-images/{unique_id}outlines_transparent.png")
    shutil.copyfile(outline_path, f"static/archived-images/{unique_id}_outlined.png")
    shutil.copyfile(temp_path, f"static/archived-images/{unique_id}_bw.png")
    
    return outline_path, transparent_path, temp_path


@app.route('/outlinefile', methods=['GET', 'POST'])
def outlinefile():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image and get the output path
            savefile, display_filename, temp_filename = outlineJ(filepath, sigma=0.33)
            return render_template('outlines.html', filename=display_filename, temp_filename=temp_filename)
    return '''
    <!doctype html>
    <title>Upload an image</title>
    <h1>Upload an image for outline processing</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

#def allowed_file(filename):
#    return '.' in filename and filename.rsplit('.', 1)[1].lower() in{'png', 'jpg', 'jpeg', 'gif'}

# SQLite database setup functions
def get_db_connection():
    """
    Establishes a connection to the SQLite database.
    """
    try:
        conn = sqlite3.connect(DATABASEF)
        conn.row_factory = sqlite3.Row
        logit("Database connection established.")
        return conn
    except sqlite3.Error as e:
        logit(f"Error establishing database connection: {e}")
        traceback.print_exc()
        return None

def create_db():
    """
    Initializes the database by creating necessary tables if they don't exist.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_text TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        logit(f"Database {DATABASEF} initialized.")
    except sqlite3.Error as e:
        logit(f"Error initializing database: {e}")
        traceback.print_exc()

def read_functions():
    """
    Reads all function texts from the database.
    """
    logit("Reading functions from database...")
    try:
        conn = get_db_connection()
        if conn is None:
            logit("Failed to establish database connection.")
            return []
        cursor = conn.cursor()
        cursor.execute('SELECT function_text FROM functions')
        functions = [row[0] for row in cursor.fetchall()]
        conn.close()
        logit("Functions retrieved from database.")
        return functions
    except sqlite3.Error as e:
        logit(f"Error reading functions: {e}")
        traceback.print_exc()
        return []

def insert_function(function_text):
    """
    Inserts a new function text into the database.
    """
    try:
        logit("Inserting function into database...")
        conn = get_db_connection()
        if conn is None:
            logit("Failed to establish database connection.")
            return
        cursor = conn.cursor()
        cursor.execute('INSERT INTO functions (function_text) VALUES (?)', (function_text,))
        conn.commit()
        conn.close()
        logit("Function inserted into database.")
    except sqlite3.Error as e:
        logit(f"Error inserting function: {e}")
        traceback.print_exc()

def insert_functions():
    """
    Inserts functions from 'con_html.txt' into the database if not already initialized.
    """
    logit("Checking if functions need to be inserted into the database...")
    try:
        conn = get_db_connection()
        if conn is None:
            logit("Failed to establish database connection.")
            return
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key='initialized'")
        result = cursor.fetchone()
        if result is None:
            logit("Initializing and inserting functions from 'con_html.txt'...")
            with open('all_html.txt', 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            # Assuming functions are separated by '\n.\n'
            segments = content.strip().split('@app')
            for segment in segments:
                cleaned_segment = segment.strip()
                if cleaned_segment:
                    cursor.execute('INSERT INTO functions (function_text) VALUES (?)', (cleaned_segment,))
            cursor.execute("INSERT INTO metadata (key, value) VALUES ('initialized', 'true')")
            conn.commit()
            logit("Functions inserted into database.")
        else:
            logit("Functions already inserted into database.")
        conn.close()
    except sqlite3.Error as e:
        logit(f"Error inserting functions into database: {e}")
        traceback.print_exc()

def get_last_function():
    """
    Retrieves the most recently inserted function from the database.
    """
    logit("Retrieving the last function from the database...")
    try:
        conn = get_db_connection()
        if conn is None:
            logit("Failed to establish database connection.")
            return None
        cursor = conn.cursor()
        cursor.execute('SELECT function_text FROM functions ORDER BY id DESC LIMIT 1')
        result = cursor.fetchone()
        conn.close()
        if result:
            logit("Last function retrieved successfully.")
            return result[0]
        else:
            logit("No functions found in the database.")
            return None
    except sqlite3.Error as e:
        logit(f"Error retrieving last function: {e}")
        traceback.print_exc()
        return None

@app.route('/index_code')
def index_code():
    """
    Renders the main index page with the latest function.
    """
    functions = get_last_function()
    return render_template('index_code.html', functions=functions)

@app.route('/save', methods=['POST'])
def save():
    """
    Saves the provided code and generates suggestions.
    """
    code = request.form['code']
    suggestions = generate_suggestions(code)
    return {'suggestions': suggestions}

def generate_suggestions(code):
    """
    Generates suggestions based on the last two words of the provided code.
    Each suggestion is approximately 400 characters long.
    """
    logit("Generating suggestions...")
    functions = read_functions()

    if not functions:
        logit("No functions available to generate suggestions.")
        return []

    # Retrieve the last line from the code
    lines = code.strip().split('\n')
    last_line = lines[-1] if lines else ''
    logit(f"Last line of code: '{last_line}'")

    # Split the last line into words and get the last two words
    words = last_line.split()
    last_two_words = ' '.join(words[-2:]) if len(words) >=2 else last_line
    logit(f"Last two words: '{last_two_words}'")

    # Function to split snippet based on last_two_words and return completion
    def split_snippet(snippet, last_two_words):
        index = snippet.rfind(last_two_words)
        if index != -1:
            completion = snippet[index + len(last_two_words):].strip()
            return completion
        return snippet.strip()

    # Search for matching snippets based on the last two words
    matching_snippets = []
    found_indices = set()  # To store indices of found snippets to avoid duplicates

    for i, snippet in enumerate(functions, start=1):
        if last_two_words in snippet:
            if i not in found_indices:
                found_indices.add(i)
                completion = split_snippet(snippet, last_two_words)
                formatted_snippet = f"<pre>{i}: {completion}</pre>"
                # Adjust the snippet length to approximately 400 characters
                if len(formatted_snippet) > 400:
                    formatted_snippet = formatted_snippet[:397] + '...'
                matching_snippets.append(formatted_snippet)
                logit(f"Added snippet {i}: {formatted_snippet}")

    # Return up to 20 suggestions, limited to 5 for demonstration purposes
    suggestions = matching_snippets[:5]
    logit(f"Generated {len(suggestions)} suggestions.")
    return suggestions

@app.route('/save_code', methods=['POST'])
def save_code():
    """
    Saves the provided code to the database.
    """
    code = request.data.decode('utf-8')
    logit(f"Received code to save: {code[:50]}...")  # Log first 50 characters for brevity
    if code:
        insert_function(code)
        return 'Code saved successfully', 200
    else:
        logit("No code provided in the request.")
        return 'No code provided in the request', 400

@app.route('/functions', methods=['GET', 'POST'])
def get_functions():
    """
    Retrieves all functions from the database and returns them as JSON.
    """
    logit("Fetching all functions from the database.")
    conn = get_db_connection()
    if conn is None:
        logit("Failed to establish database connection.")
        return jsonify([]), 500
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM functions')
    functions = cursor.fetchall()
    conn.close()
    logit(f"Retrieved {len(functions)} functions from the database.")
    return jsonify([dict(ix) for ix in functions])

@app.route('/functions/<int:id>', methods=['PUT', 'GET'])
def update_response(id):
    """
    Updates the function text for a given function ID.
    """
    new_response = request.json.get('function_text')
    logit(f"Updating function ID {id} with new text.")
    if not new_response:
        logit("No new function text provided.")
        return jsonify({'status': 'failure', 'message': 'No function text provided'}), 400
    try:
        conn = get_db_connection()
        if conn is None:
            logit("Failed to establish database connection.")
            return jsonify({'status': 'failure', 'message': 'Database connection failed'}), 500
        cursor = conn.cursor()
        cursor.execute('UPDATE functions SET function_text = ? WHERE id = ?', (new_response, id))
        conn.commit()
        conn.close()
        logit(f"Function ID {id} updated successfully.")
        return jsonify({'status': 'success', 'message': 'Function updated successfully'})
    except sqlite3.Error as e:
        logit(f"Error updating function ID {id}: {e}")
        traceback.print_exc()
        return jsonify({'status': 'failure', 'message': 'Error updating function'}), 500

@app.route('/view_functions', methods=['GET', 'POST'])
def view_functions():
    """
    Renders a page to view all functions.
    """
    logit("Rendering view_functions page.")
    conn = get_db_connection()
    if conn is None:
        logit("Failed to establish database connection.")
        return render_template('view_functions.html', data=[])
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM functions')
    data = cursor.fetchall()
    conn.close()
    logit(f"Retrieved {len(data)} functions for viewing.")
    return render_template('view_functions.html', data=data)

@app.route('/update_function', methods=['POST', 'GET'])
def update_function():
    """
    Updates a specific function based on form data and redirects to the view page.
    """
    id = request.form.get('id')
    new_function_text = request.form.get('function_text')
    logit(f"Received update for function ID {id}.")
    if not id or not new_function_text:
        logit("Missing function ID or new function text in the request.")
        return redirect(url_for('view_functions'))
    try:
        conn = get_db_connection()
        if conn is None:
            logit("Failed to establish database connection.")
            return redirect(url_for('view_functions'))
        cursor = conn.cursor()
        cursor.execute('UPDATE functions SET function_text = ? WHERE id = ?', (new_function_text, id))
        conn.commit()
        conn.close()
        logit(f"Function ID {id} updated successfully via form.")
        return redirect(url_for('view_functions'))
    except sqlite3.Error as e:
        logit(f"Error updating function ID {id}: {e}")
        traceback.print_exc()
        return redirect(url_for('view_functions'))

def get_suggestions(search_term):
    try:
        # Create a database connection
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()

        # Create table
        c.execute('''CREATE TABLE IF NOT EXISTS dialogue
                     (id INTEGER PRIMARY KEY,
                      search_term TEXT,
                      ChatGPT_PAIR TEXT,
                      ChatGPT_PAIRb BLOB
                      )''')
        conn.commit()

        # Perform operations
        cnt = 0
        DATA = set()
        INDEX = '----SplitHere------'
        with open("app.py", "r") as data:
            Lines = data.read()
            lines = Lines.replace(search_term, INDEX + search_term)
            lines = lines.split(INDEX)
            for line in lines:
                if search_term in line:
                    cnt += 1
                    DATA.add(f'{line[:1200]}')
                    # Insert dialogue pair into the table
                    c.execute("INSERT INTO dialogue (search_term, ChatGPT_PAIR, ChatGPT_PAIRb) VALUES (?, ?, ?)",
                              (search_term, line, line.encode('utf-8')))
                    conn.commit()

        # Close the database connection
        conn.close()
        return DATA
    except Exception as e:
        logit(f"An error occurred: {e}")
        return set()

@app.route('/search_file', methods=['GET', 'POST'])
def search_file():
    search_term = request.args.get('q', '') if request.method == 'GET' else request.form.get('q', '')
    if search_term:
        data = get_suggestions(search_term)
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Search Results</title>

        </head>
        <body>
        <a style="color:navy;font-size:24px;" href="/search_file">Search Again</a><br>
        <a style="color:navy;font-size:24px;" href="/">Home</a>
            <h1>Search Results for "{{ search_term }}"</h1>
            {% for item in data %}
                <div style="border-bottom: 1px solid #ccc; padding: 10px;">
                    <pre style="color:navy;font-size:24px;">{{ item }}</pre>
                </div>
            {% endfor %}
            
        </body>
        </html>
        ''', search_term=search_term, data=data)
    else:
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enter Search Term</title>
        </head>
        <body>
            <h1>Enter a Search Term</h1>
            <form action="/search_file" method="post">
                <input type="text" name="q" placeholder="Enter search term">
                <input type="submit" value="Search">
            </form>
        </body>
        </html>
        ''')
@app.route('/convert-images')
def convert_images_route():
    try:
        convert_images()
        return redirect(url_for('mk_videos'))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/create-video')
def create_video_route():
    try:
        createvideo()
        return redirect(url_for('mk_videos'))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/refresh-video')
def refresh_video_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'refresh_video.py'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/best-flipbook')
def best_flipbook_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'Best_FlipBook'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/diagonal-transition')
def diagonal_transition_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'diagonal_transition'], check=True)
        video_path = 'static/temp_exp/diagonal1.mp4'
        add_title(video_path, hex_color="#A52A2A")
        return redirect(url_for('mk_videos'))
    
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/blendem')
def blendem_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'blendem'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/slide')
def slide_route():
    try:
        subprocess.run(['/bin/bash', 'slide'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/zoomx4')
def zoomx4_route():
    try:
        subprocess.run(['/bin/bash', 'zoomX4'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500
    
@app.route('/zoomy4')
def zoomy4_route():
    try:
        subprocess.run(['/bin/bash', 'zoomY4'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/vertical-scroll')
def vertical_scroll_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'vertical_scroll'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500
@app.route('/zoom_each')
def zoom_each_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'refresh_video.py'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500
@app.route('/add-title')
def add_title_route():
    try:
        video_path = 'static/temp_exp/diagonal1.mp4'
        add_title(video_path, hex_color="#A52A2A")
        return redirect(url_for('mk_videos'))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/join-video')
def join_video_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'joinvid'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/refresh-all')
def refresh_all_route():
    try:
        # Call each route sequentially
        convert_and_resize_images_route()
        create_video_route()
        refresh_video_route()
        best_flipbook_route()
        diagonal_transition_route()
        blendem_route()
        slide_route()
        zoomx4_route()
        vertical_scroll_route()
        add_title_route()
        join_video_route()
        resize_mp4_route

        return redirect(url_for('create_video'))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/convert_and_resize_images')
def convert_and_resize_images_route():
    img_dir = 'static/archived-store'

    # Ensure the directory exists
    if not os.path.exists(img_dir):
        return "Directory not found", 404

    # Get all files in the directory and sort by creation time
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
    image_files = sorted(image_files, key=lambda x: os.path.getctime(os.path.join(img_dir, x)), reverse=True)

    

    # Iterate over sorted images
    for file in image_files:
        img_file_path = os.path.join(img_dir, file)

        # Convert to JPG if PNG and set the new path
        if file.lower().endswith('.png'):
            new_file_path = os.path.splitext(img_file_path)[0] + '.jpg'
        else:
            new_file_path = img_file_path

        try:
            # Open the image
            with Image.open(img_file_path) as img:
                # Resize image to height of 768 while maintaining aspect ratio
                width, height = img.size
                new_height = 768
                new_width = int((new_height / height) * width)
                resized_img = img.resize((new_width, new_height))

                # Center crop to 512x768
                left = (new_width - 512) / 2
                top = 0
                right = (new_width + 512) / 2
                bottom = 768
                cropped_img = resized_img.crop((left, top, right, bottom))

                # Convert to RGB (only for PNGs)
                if file.lower().endswith('.png'):
                    cropped_img = cropped_img.convert('RGB')

                # Save the resulting image as JPG
                cropped_img.save(new_file_path, 'JPEG')

        except Exception as e:
            continue

    return redirect(url_for('conversion_complete'))
@app.route('/conversion_complete')
def conversion_complete():
    return "All PNG and JPG files have been converted, resized to 512x768, and saved."

@app.route('/base')
def base():
    return render_template('base_1.html')

# Step 1: Resize videos and save as basename_512x768.mp4
def resize_videos(directory, target_size=(512, 768)):
    for filename in os.listdir(directory):
        if filename.endswith("X.mp4") and not filename.endswith("_512x768.mp4"):
            filepath = os.path.join(directory, filename)

            # Load the video file
            video_clip = VideoFileClip(filepath)

            # Resize the video to the target size (512x768)
            resized_clip = video_clip.resize(newsize=target_size)

            # Save the resized video with the new name basename_512x768.mp4
            new_filename = os.path.splitext(filename)[0] + "_512x768.mp4"
            resized_filepath = os.path.join(directory, new_filename)
            
            # Write the resized video to a file
            resized_clip.write_videofile(resized_filepath)

            # Close the clip to release resources
            video_clip.close()

# Step 2: Concatenate all the resized videos
def concatenate_resized_videos(directory, output_file):
    video_clips = []

    # Look for all files that match *_512x768.mp4
    for filename in os.listdir(directory):
        if filename.endswith("_512x768.mp4"):
            filepath = os.path.join(directory, filename)

            # Load the resized video file
            video_clip = VideoFileClip(filepath)

            # Add to the list of clips to concatenate
            video_clips.append(video_clip)

    # Concatenate all resized video clips
    if video_clips:
        final_clip = concatenate_videoclips(video_clips, method="compose")

        # Write the final concatenated clip to the output file
        final_clip.write_videofile(output_file)

        # Close the clips to release resources
        for clip in video_clips:
            clip.close()
@app.route('/resize_mp4')
def resize_mp4_route():

    # Directory containing the original *.mp4 files
    input_directory = "/home/jack/Desktop/Flask_Make_Art/static/temp_exp"

    # Output file name for the concatenated video
    output_file = "/home/jack/Desktop/Flask_Make_Art/static/temp_exp/all_asset_videos.mp4"

    # Step 1: Resize and save videos
    resize_videos(input_directory)

    # Step 2: Concatenate resized videos
    concatenate_resized_videos(input_directory, output_file)     
    return redirect(url_for('display_resources'))








archived_images_dir = 'static/archived-store'

def resize_and_crop_image(image_path):
    """Resize image to height 768 keeping aspect ratio, then center-crop to 512x768."""
    try:
        print(f"Processing image: {image_path}")
        with Image.open(image_path) as img:
            # Resize image to height 768 while maintaining the aspect ratio
            width, height = img.size
            aspect_ratio = width / height
            new_height = 768
            new_width = int(aspect_ratio * new_height)

            print(f"Original size: {width}x{height}, Resizing to: {new_width}x{new_height}")

            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Calculate cropping box to center-crop the image to 512x768
            left = (new_width - 512) / 2
            top = 0  # Since height is already 768, no need to crop vertically
            right = (new_width + 512) / 2
            bottom = 768

            print(f"Cropping coordinates: left={left}, top={top}, right={right}, bottom={bottom}")

            img = img.crop((left, top, right, bottom))

            # Save the cropped image as JPG
            new_image_path = os.path.splitext(image_path)[0] + '.jpg'
            img.convert('RGB').save(new_image_path, 'JPEG')

            print(f"Saved resized and cropped image: {new_image_path}")

            # Optionally, remove the original PNG file if it exists
            if image_path.lower().endswith('.png'):
                os.remove(image_path)
                print(f"Removed original PNG file: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


@app.route('/size_and_format_images')
def size_and_format_images_route():
    """Resize and format PNG/JPG images in the specified directory."""
    print("Starting image resizing and formatting...")
    for image_path in glob.glob(os.path.join(archived_images_dir, '*')):
        if image_path.lower().endswith(('.png', '.jpg')):
            print(f"Resizing image: {image_path}")
            resize_and_crop_image(image_path)
    print("Image resizing completed.")
    return redirect(url_for('img_processing_route'))


@app.route('/clean_storage', methods=['GET', 'POST'])
def clean_storage_route():
    # Resize and format images before processing
    size_and_format_images_route()

    if request.method == 'POST':
        # Get selected images from form
        selected_images = request.form.getlist('selected_images')

        if selected_images:
            # Generate a unique video file name
            unique_id = str(uuid.uuid4())
            video_filename = os.path.join('static/image-archives', f'{unique_id}.mp4')

            # Ensure the directory for video exists
            if not os.path.exists('static/image-archives'):
                os.makedirs('static/image-archives')

            # Get dimensions from the first image to set video size
            first_image_path = os.path.join(archived_images_dir, selected_images[0])
            first_image = cv2.imread(first_image_path)
            height, width, layers = first_image.shape

            # Define the video codec and create VideoWriter object
            video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

            # Write each selected image to the video
            for image in selected_images:
                image_path = os.path.join(archived_images_dir, image)
                img = cv2.imread(image_path)
                video.write(img)

            # Release the VideoWriter object after writing the video
            video.release()

            # Remove selected images from storage
            for image in selected_images:
                image_path = os.path.join(archived_images_dir, image)
                if os.path.exists(image_path):
                    os.remove(image_path)

        return redirect(url_for('img_processing_route'))

    # Fetch all images (JPG and PNG) in the archived directory
    images = [os.path.basename(img) for img in glob.glob(os.path.join(archived_images_dir, '*.jpg'))]
    images.extend([os.path.basename(img) for img in glob.glob(os.path.join(archived_images_dir, '*.png'))])

    # Sort images by creation time
    images = sorted(images, key=lambda x: os.path.getctime(os.path.join(archived_images_dir, x)), reverse=True)

    return render_template('img_processing.html', images=images)

@app.route('/get_videos', methods=['GET', 'POST'])
def get_videos():
    video_files = glob.glob("static/video_history/*.mp4")
    return render_template("get_videos.html", video_files=video_files)
@app.route('/upload_form')
def upload_form():
    image_directory = 'static/archived-store/'  # Directory containing the images
    images_list = []

    # Retrieve all image files from the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg') or f.endswith('.png')]
    # List by date of creation reverse
    image_files = sorted(image_files, key=lambda x: os.path.getctime(os.path.join(image_directory, x)), reverse=True)
    return render_template('upload_form.html', video_url=None, video_type=None, image_files=image_files)


@app.route('/process_selected_images', methods=['POST'])
def process_selected_images():
    # Check if any images were uploaded
    bg_image_uploaded = request.files.get('bg_image')
    fg_image_uploaded = request.files.get('fg_image')

    # Check if any existing images were selected
    bg_image_selected = request.form.get('bg_image_selected')
    fg_image_selected = request.form.get('fg_image_selected')

    if bg_image_uploaded and bg_image_uploaded.filename != '':
        bg_filename = 'background.png'
        bg_image_uploaded.save(os.path.join('static', bg_filename))
        bg_file_path = os.path.abspath(os.path.join('static', bg_filename))
    elif bg_image_selected:
        bg_file_path = os.path.abspath(os.path.join('static/archived-store', bg_image_selected))
    else:
        return redirect(url_for('upload_form'))

    if fg_image_uploaded and fg_image_uploaded.filename != '':
        fg_filename = 'foreground.png'
        fg_image_uploaded.save(os.path.join('static', fg_filename))
        fg_file_path = os.path.abspath(os.path.join('static', fg_filename))
    elif fg_image_selected:
        fg_file_path = os.path.abspath(os.path.join('static/archived-store', fg_image_selected))
    else:
        return redirect(url_for('upload_form'))

    # Apply zoom effect and get the processed image list
    images_list = zoom_effect(bg_file_path, fg_file_path)

    if not os.path.exists('static/overlay_zooms/title'):
        os.makedirs('static/overlay_zooms/title')

    output_mp4_file = 'static/overlay_zooms/title/title_video.mp4'
    frames_per_second = 30
    create_mp4_from_images(images_list, output_mp4_file, frames_per_second)

    # Create a timestamped copy of the output MP4
    file_bytime = time.strftime("%Y%m%d-%H%M%S") + ".mp4"
    shutil.copy(output_mp4_file, 'static/overlay_zooms/title/' + file_bytime)
    video_url = url_for('static', filename='overlay_zooms/title/' + file_bytime)

    return render_template('upload.html', video_url=video_url, video_type="title")

# Route to process all images in a directory (main video)@app.route('/zoom_all')
@app.route('/zoom_all')
def process_directory():
    image_directory = 'static/archived-store/'  # Directory containing the images
    images_list = []

    # Retrieve all image files from the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg') or f.endswith('.png')]
    #list by date of creation reverse
    image_files = sorted(image_files, key=lambda x: os.path.getctime(os.path.join(image_directory, x)), reverse=True)

    # Ensure there are at least two images to work with
    if len(image_files) < 2:
        logit(f"Not enough images in {image_directory} to process.")
        return render_template('upload.html', error="Not enough images to apply zoom effect.")
    
    # Iterate over image pairs
    for i in range(0, len(image_files) - 1, 2):
        # Define image paths for the current pair
        image_path1 = os.path.join(image_directory, image_files[i])
        image_path2 = os.path.join(image_directory, image_files[i + 1])

        # Apply zoom effect with two different images
        images = zoom_effect(image_path1, image_path2)
        images_list.extend(images)

        # Logging the processed images
        logit(f"Processed images: {image_files[i]} and {image_files[i + 1]}")

    # Ensure the output directory exists
    if not os.path.exists('static/overlay_zooms/main'):
        os.makedirs('static/overlay_zooms/main')

    # Generate MP4 file from processed images
    output_mp4_file = 'static/overlay_zooms/main/main_video.mp4'
    frames_per_second = 30
    create_mp4_from_images(images_list, output_mp4_file, frames_per_second)

    # Save the output with a timestamp
    file_bytime = time.strftime("%Y%m%d-%H%M%S") + ".mp4"
    shutil.copy(output_mp4_file, 'static/overlay_zooms/main/' + file_bytime)
    video_url = url_for('static', filename='overlay_zooms/main/' + file_bytime)

    # Copy the video to a temporary directory
    dst = 'static/temp_exp/vopt.mp4'
    src = output_mp4_file
    shutil.copy(src, dst)

    return render_template('upload.html', video_url=video_url, video_type="main")


# Route to concatenate title and main videos
@app.route('/concatenate_videos')
def concatenate_videos():
    title_video = 'static/overlay_zooms/title/title_video.mp4'
    main_video = 'static/overlay_zooms/main/main_video.mp4'

    if not os.path.exists(title_video) or not os.path.exists(main_video):
        return "Both title and main videos must exist to concatenate."

    title_clip = VideoFileClip(title_video)
    main_clip = VideoFileClip(main_video)

    final_clip = concatenate_videoclips([title_clip, main_clip])

    output_file = 'static/overlay_zooms/final/final_video.mp4'
    if not os.path.exists('static/overlay_zooms/final'):
        os.makedirs('static/overlay_zooms/final')

    final_clip.write_videofile(output_file, codec="libx264", fps=30)

    file_bytime = time.strftime("%Y%m%d-%H%M%S") + ".mp4"
    shutil.copy(output_file, 'static/overlay_zooms/final/' + file_bytime)

    video_url = url_for('static', filename='overlay_zooms/final/' + file_bytime)
    shutil.copy(output_file, 'static/temp_exp/finalX.mp4')
    #return render_template('upload.html', video_url=video_url, video_type="final")
    return redirect(url_for('mk_videos'))



# Zoom effect function (unchanged)
def zoom_effect(bg_file, fg_file):
    # Open background and foreground images, convert them to RGBA for transparency
    bg = Image.open(bg_file).convert('RGBA')
    fg = Image.open(fg_file).convert('RGBA')
    
    # Get size of the background image
    SIZE = bg.size
    
    # Resize background and foreground images to the same size for consistency
    bg = bg.resize(SIZE, resample=Image.Resampling.LANCZOS)
    fg = fg.resize(SIZE, resample=Image.Resampling.LANCZOS)

    result_images = []

    # Generate zoom effect by resizing the foreground progressively
    for i in range(200):
        # Calculate progressive size scaling for foreground
        size = (int(fg.width * (i + 1) / 200), int(fg.height * (i + 1) / 200))
        
        # Resize foreground with progressive zoom
        fg_resized = fg.resize(size, resample=Image.Resampling.LANCZOS)
        
        # Apply gradual transparency (alpha) to foreground
        fg_resized.putalpha(int((i + 1) * 255 / 200))

        # Create a copy of the background image to composite the foreground onto
        result = bg.copy()

        # Calculate position to center the resized foreground
        x = int((bg.width - fg_resized.width) / 2)
        y = int((bg.height - fg_resized.height) / 2)

        # Alpha composite the resized foreground onto the background
        result.alpha_composite(fg_resized, (x, y))

        # Add the result to the list of images
        result_images.append(result)

    return result_images
# Function to create MP4 from images (unchanged)
def create_mp4_from_images(images_list, output_file, fps):
    #latest change
    #images_list = sorted(images_list, key=lambda x: os.path.getctime(os.path.join('static/archived_store', x)), reverse=True)
    image_arrays = [np.array(image) for image in images_list]
    clip = ImageSequenceClip(image_arrays, fps=fps)
    clip.write_videofile(output_file, codec="libx264", fps=fps)

@app.route('/archive_images')
def archive_images_route():
    source_dir = os.path.join('static', 'masks')
    dest_dir = os.path.join('static', 'archived-images/')

    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Copy each image from source to destination
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions as needed
            src_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            shutil.copy(src_file, dest_file)

    return redirect(url_for('select_images'))  # Redirect to an appropriate page

@app.route('/view_masks')
def view_masks():
    masks = glob.glob('static/masks/*.jpg')
    masks = sorted(masks, key=os.path.getmtime, reverse=True)
    filenames = [os.path.basename(mask) for mask in masks]
    mask_data = zip(masks, filenames)
    return render_template('view_masks.html', mask_data=mask_data)
@app.route('/delete_mask', methods=['POST', 'GET'])
def delete_mask():
    mask_path = request.form.get('mask_path')
    if mask_path:
        try:
            os.remove(mask_path)
            logit(f"Deleted mask: {mask_path}")
        except Exception as e:
            logit(f"Error deleting mask: {e}")
    return redirect(url_for('view_masks'))


# Paths

TITLE_DIR = 'static/overlay_zooms/title/'
MAIN_DIR = 'static/temp_exp/'
OUTPUT_PATH = 'static/temp_exp/titled.mp4'
RESIZED_TITLE_PATH = 'static/temp_exp/resized_title.mp4'
RESIZED_MAIN_PATH = 'static/temp_exp/resized_main.mp4'

@app.route('/title', methods=['GET', 'POST'])
def title_route():
    if request.method == 'POST':
        # Get selected files
        title_file = request.form.get('title_file')
        main_file = request.form.get('main_file')
        
        if title_file and main_file:
            # Construct file paths
            title_path = os.path.join(TITLE_DIR, title_file)
            main_path = os.path.join(MAIN_DIR, main_file)
            
            # Resize the videos to match dimensions
            resize_videos(title_path, main_path)
            
            # Concatenate the videos
            concatenate_title_video()
            
            return redirect(url_for('title_route', result='success'))

    # List files in the directories
    title_files = [f for f in os.listdir(TITLE_DIR) if f.endswith('.mp4')]
    main_files = [f for f in os.listdir(MAIN_DIR) if f.endswith('.mp4')]
    
    return render_template('title.html', title_files=title_files, main_files=main_files)

def resize_videos(title_path, main_path):
    # Define common dimensions
    width = 512
    height = 768
    
    # Resize the title video
    command_title = [
        'ffmpeg', '-hide_banner',
        '-i', title_path,
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-y', RESIZED_TITLE_PATH
    ]
    
    # Resize the main video
    command_main = [
        'ffmpeg', '-hide_banner',
        '-i', main_path,
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-y', RESIZED_MAIN_PATH
    ]
    
    # Execute the resize commands
    try:
        subprocess.run(command_title, check=True)
        subprocess.run(command_main, check=True)
        print("Resizing successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during resizing: {e}")
@app.route('/concat_videos')
def concat_videos():
    # Define the file paths
    output_file = 'static/temp_exp/OUTPUT_PATHX.mp4'
    title_path = 'static/temp_exp/resized_title.mp4'
    main_path = 'static/temp_exp/resized_main.mp4'
    mp3_directory = 'static/mp3s'

    # Check if input files exist
    if not os.path.exists(title_path):
        print(f"Error: Title video '{title_path}' not found.")
        return "Title video not found", 400
    if not os.path.exists(main_path):
        print(f"Error: Main video '{main_path}' not found.")
        return "Main video not found", 400

    # Pick a random MP3 file from static/mp3s directory
    try:
        mp3_files = [f for f in os.listdir(mp3_directory) if f.endswith('.mp3')]
        if not mp3_files:
            print("No MP3 files found in 'static/mp3s'.")
            return "No MP3 files available", 400
        random_mp3 = os.path.join(mp3_directory, random.choice(mp3_files))
        print(f"Selected random MP3: {random_mp3}")
    except Exception as e:
        print(f"Error selecting random MP3: {e}")
        return f"Error selecting MP3: {e}", 500

    # Construct the ffmpeg command with audio
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'info',  # Enable FFmpeg logging for debugging
        '-i', title_path,
        '-i', main_path,
        '-i', random_mp3,  # Add the randomly selected MP3
        '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[outv]',
        '-map', '[outv]',  # Video output stream
        '-map', '2:a',     # Map the audio from the random MP3
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-shortest',        # Use shortest flag to end when either video or audio finishes
        '-y', output_file   # Output file
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print("Video concatenation and audio addition successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during concatenation: {e}")
        return f"Error during concatenation: {e}", 500
    
    return redirect(url_for('mk_videos'))
def concatenate_title_video():
    # Define the file paths
    output_file = 'temp_exp/OUTPUT_PATHX.mp4'
    title_path = 'static/temp_exp/resized_title.mp4'
    main_path = 'static/temp_exp/resized_main.mp4'
    # Construct the ffmpeg command
    command = [
        'ffmpeg', '-hide_banner',
        '-i', title_path,
        '-i', main_path,
        '-filter_complex',
        '[0:v][1:v]concat=n=2:v=1:a=0[outv];[1:a]aformat=sample_rates=44100[aout]',
        '-map', '[outv]',
        '-map', '[aout]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-shortest',
        '-y', output_file
    ]
    
    # Execute the command
    try:
        subprocess.run(command, check=True)
        print("Concatenation successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during concatenation: {e}")
def bak(filename):
    try:
        # Get the current date and time
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Generate a backup filename with the timestamp
        backup_filename = f"backup_{timestamp}.py"
        
        # Copy the file to the new backup filename
        shutil.copy(filename, backup_filename)
        
        print(f"File '{filename}' successfully backed up as '{backup_filename}'.")
    except Exception as e:
        print(f"An error occurred while backing up the file: {e}")
@app.route('/moviepy')
def moviepy_route():
    return render_template('moviepy_info.html')

@app.route('/flask_info')
def flask_info_route():
    return render_template('flask_info.html')

@app.route('/moviepy_fx_info')
def moviepy_fx_route():
    return render_template('moviepy_fx.html')

@app.route('/PIL_info')
def PIL_info_route():
    return render_template('PIL_info.html')


if __name__ == '__main__':

    bak('app.py')
    #remove log file
    try:
        log_file = "static/app_log.txt"
        dataout = open(log_file, 'r').read()
        if len(dataout) > 100000:
            open(log_file, 'w').close()
        logit("Log file cleared.")
    except: pass
    directory = 'static/TEXT'
    load_txt_files(directory)
    app.run(debug=True, host='0.0.0.0', port=5000)
