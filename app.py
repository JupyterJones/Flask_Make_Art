#!/home/jack/miniconda3/envs/cloned_base/bin/python
import os
import random
import glob
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, flash
from PIL import Image, ImageOps
from moviepy.editor import VideoFileClip
from datetime import datetime
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
#app.config['UPLOAD_FOLDER'] = 'static/KLING'
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
        timestr = datetime.now().strftime('%A_%b-%d-%Y_%H-%M-%S')
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
        with open("exp_log.txt", "a") as file:
            # Write the log message to the file
            file.write(log_message)

            # Print the log message to the console
            print(log_message)

    except Exception as e:
        # If an exception occurs during logging, print an error message
        print(f"Error occurred while logging: {e}")

logit("App started")

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
    random.shuffle(image_paths)
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
'''
@app.route('/select_mask_image', methods=['GET', 'POST'])
def select_mask_image():
    if request.method == 'POST':
        selected_image = request.form.get('selected_image')
        return redirect(url_for('choose_mask', image_path=selected_image))

    image_paths = get_image_paths()
    return render_template('select_mask_image.html', image_paths=image_paths)

@app.route('/choose_mask/<image_path>', methods=['GET', 'POST'])
def choose_mask(image_path):
    if request.method == 'POST':
        mask_type = request.form.get('mask_type')
        if mask_type == 'grayscale':
            mask_path = convert_to_grayscale(image_path)
        elif mask_type == 'binary':
            mask_path = convert_to_binary(image_path)
        else:
            return "Invalid mask type selected."
        return render_template('display_images_exp.html', image_paths=[image_path], mask_path=mask_path, opacity=0.5)
    
    return render_template('mask_type.html', image_path=image_path)
'''
@app.route('/edit_mask', methods=['POST'])
def edit_mask():
    image_paths = request.form.getlist('image_paths')
    mask_path = request.form.get('mask_path')
    opacity = float(request.form.get('opacity', 0.5))
    return render_template('display_images_exp.html', image_paths=image_paths, mask_path=mask_path, opacity=opacity)

@app.route('/store_result', methods=['POST'])
def store_result():
    result_image_path = request.form.get('result_image')
    unique_id = datetime.now().strftime('%Y%m%d%H%M%S')
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
    try:
        # Run the script using subprocess
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'refresh_video.py'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'Best_FlipBook'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'diagonal_transition'], check=True)        
        return redirect(url_for('index'))
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"

@app.route('/display_resources', methods=['POST','GET'])
def display_resources():
    images = glob.glob('static/archived-images/*.jpg')
    image_directory = app.config['UPLOAD_FOLDER']
    logit(f"Image directory: {image_directory}")
    image_paths = glob.glob('static/archived-images/*.jpg')
    #list by date last image first
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    return render_template('display_resources_exp.html', image_paths=image_paths)

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

    unique_id = datetime.now().strftime('%Y%m%d%H%M%S')
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
    return render_template('upload.html')

@app.route('/upload', methods=['POST', 'GET'])
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
    return redirect(url_for('index'))

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

'''
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

    # Blend the images
    for i in range(3):  # For each color channel
        background_rgba[:, :, i] = (foreground_rgb[:, :, i] * alpha_channel + background_rgba[:, :, i] * (1 - alpha_channel)).astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, background_rgba)

# Generate a unique filename
    unique_filename = 'static/archived-images/' + str(uuid.uuid4()) + '.jpg'

    # Optionally convert the composite to a JPG
    im = Image.open(output_path).convert('RGB')
    im.save(unique_filename, quality=95)

    print(f"Composite image saved to: {unique_filename}")
    
    return unique_filename
    #return output_path
'''
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
# Example usage
# Paths to the input images and output composite image
# use glob to get random choice of image
'''
foreground_image_path = random.choice(glob.glob("static/archived-images/*.jpg"))
background_image_path = random.choice(glob.glob("static/archived-images/*.jpg"))
#foreground_image_path = 'static/archived-images/face.jpg'
feathered_image_path = 'static/archived-images/feathered_face.png'
#background_image_path = 'static/archived-images/background.jpg'
#use uuid to create a unique name for the composite image
output_composite_path = 'static/archived-images/composite_image' + str(uuid.uuid4()) + '.png'
#output_composite_path = 'static/archived-images/composite_image.png'

# Create a feathered PNG image from the detected face
create_feathered_image(foreground_image_path, feathered_image_path)

# Overlay the feathered image on the background
overlay_feathered_on_background(feathered_image_path, background_image_path, output_composite_path)
'''

if __name__ == '__main__':
    app.run(debug=True)
