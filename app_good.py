#!/home/jack/Desktop/Flask_Make_Art/flask_env/bin/python
import os
import random
import glob
from flask import Flask, request, render_template, request, redirect, url_for, send_from_directory, send_file, flash, jsonify
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter

from moviepy.editor import VideoFileClip

import datetime
import inspect
import subprocess
import shutil
from werkzeug.utils import secure_filename
import numpy as np
import yt_dlp
import dlib
import cv2
from PIL import Image
import glob
import subprocess
import string
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/archived-images'
app.config['MASK_FOLDER'] = 'static/archived-masks'
app.config['STORE_FOLDER'] = 'static/archived-store'
#app.config['STORE_FOLDER'] = 'static/KLING'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB
# Directory to save downloaded videos and extracted images
DOWNLOAD_FOLDER = 'static/downloads'
ARCHIVED_IMAGES_FOLDER = 'static/archived-images'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['ARCHIVED_IMAGES_FOLDER'] = ARCHIVED_IMAGES_FOLDER

# Ensure the directories exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
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
    return render_template('mk_mask.html')

@app.route('/create_mask', methods=['POST'])
def create_mask():
    # Get input values from the form
    x = int(request.form.get('x', 0))
    y = int(request.form.get('y', 0))
    size = int(request.form.get('size', 50))+20
    
    feather = int(request.form.get('feather', 20))
    aspect = int(request.form.get('aspect', 0))
    
    # Calculate width and height based on aspect
    if aspect > 0:
        width = size
        height = size + aspect
    elif aspect < 0:
        width = size - abs(aspect)
        height = size
    else:
        width, height = size, size

    # Create a black background image
    background = Image.new('RGBA', (512, 768), (0, 0, 0, 255))
    
    # Create a white circle with the specified size and aspect ratio
    circle = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, width, height), fill=(255, 255, 255, 255))

    # Apply feathering
    circle = circle.filter(ImageFilter.GaussianBlur(feather))

    # Calculate position to paste the circle (centered by default)
    paste_position = (256 + x - width // 2, 384 + y - height // 2)
    background.paste(circle, paste_position, circle)
    background = background.convert('RGB')
    
    background = background.filter(ImageFilter.GaussianBlur(30))
    # Save the result
    mask_path = f'static/archived-images/mask_{x}_{y}_{size}_{feather}_{aspect}.jpg'
    background.save(mask_path)

    return send_file(mask_path, as_attachment=True)
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
    images = os.listdir(app.config['PUBLISH_FOLDER'])
    
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
        image_path = os.path.join(app.config['PUBLISH_FOLDER'], image_file)
        image = Image.open(image_path)

        # Draw the text on the image
        draw = ImageDraw.Draw(image)
        draw.text(position, text, font=font, fill=color)

        # Save the temporary image for preview
        temp_image_path = os.path.join(app.config['TEMP_FOLDER'], 'temp-image.png')
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

logit("App started")
def readlog():
    log_file_path = 'static/app_log.txt'    
    with open(log_file_path, "r") as Input:
        logdata = Input.read()
    # print last entry
    logdata = logdata.split("\n")
    return logdata

logdata = readlog()
logit("This is a DEBUG message for mylog.py" + str(logdata))

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
    return render_template('index_exp.html', image_paths=image_paths)

def load_images(image_directory):
    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths.extend(glob.glob(os.path.join(image_directory, f'*.{ext}')))
    #random.shuffle(image_paths)
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    return image_paths[:3]

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
    logit(f"Loaded images: {image_paths}")
    return render_template('display_images_exp.html', image_paths=image_paths, mask_path=None, opacity=0.5)

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
        return f"An error occurred: {e}"

@app.route('/refresh-video')
def refresh_video():
    convert_images()
    try:
        # Run the script using subprocess
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'refresh_video.py'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'Best_FlipBook'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'diagonal_transition'], check=True)        
        return redirect(url_for('index'))
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"

@app.route('/display_resources', methods=['POST', 'GET'])
def display_resources():
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
        return redirect(url_for('clean_storage'))
    
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
    
    #get_image_paths = lambda: [os.path.join(app.config['MASK_FOLDER'], f) for f in os.listdir(app.config['MASK_FOLDER'])]
    image_paths = get_image_paths()
    return render_template('select_mask_image.html', image_paths=image_paths)

#get_image_paths = lambda: [os.path.join(app.config['MASK_FOLDER'], f) for f in os.listdir(app.config['MASK_FOLDER'])]

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
'''
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        
        return redirect(request.url)

    file = request.files['file']
    logit(f"XXXXXXXX {file}")

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
    return redirect(url_for('index'))
'''
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

            # Create a feathered PNG image from the detected face
            feathered_image_path = create_feathered_image(save_path, 'static/archived-images/feathered_face.png')

            # Overlay the feathered image on the background
            background_image_path = random.choice(glob.glob("static/archived-images/*.jpg"))
            output_composite_path = overlay_feathered_on_background(feathered_image_path, background_image_path, 'static/archived-images/composite_image.png')

            return render_template('face_detect.html', feathered_image=feathered_image_path, composite_image=output_composite_path)

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
        logit(post[3])# Limit to last 4 posts
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
        query = f"SELECT ROWID, title, content, image, video_filename FROM post WHERE {where_clause}"
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
            background-color: #f8f9fa;
            padding: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
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
    <pre id="generated_text"></pre>
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
@app.route('/clean_storage', methods=['GET', 'POST'])
def clean_storage():
    if request.method == 'POST':
        # Get selected images
        selected_images = request.form.getlist('selected_images')
        
        # Remove selected images
        for image in selected_images:
            image_path = os.path.join(archived_images_dir, image)
            if os.path.exists(image_path):
                os.remove(image_path)
        
        return redirect(url_for('clean_storage'))
    
    # Get list of images in the directory
    images = [os.path.basename(img) for img in glob.glob(os.path.join(archived_images_dir, '*.jpg'))]
    
    # Use `extend` to add PNG images to the list
    images.extend([os.path.basename(img) for img in glob.glob(os.path.join(archived_images_dir, '*.png'))])
    
    # Sort images by modification time
    images = sorted(images, key=lambda x: os.path.getmtime(os.path.join(archived_images_dir, x)), reverse=True) 
    
    return render_template('clean_storage.html', images=images)
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
    print("Current Directory:", PATH)
    full_path = os.path.join(PATH, directory_path, filename)

    print("Requested Filename:", filename)
    print("Directory Path:", directory_path)
    print("Full Path:", full_path)

    # Print the list of files in the directory for debugging purposes
    print("Files in Directory:", os.listdir(directory_path))

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

if __name__ == '__main__':
    #remove log file
    log_file = "static/app_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)
    directory = 'static/TEXT'
    load_txt_files(directory)
    app.run(debug=True, host='0.0.0.0', port=5000)
