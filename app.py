#!/home/jack/miniconda3/envs/cloned_base/bin/python
import os
import random
import glob
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, flash
from PIL import Image, ImageOps
from datetime import datetime
import inspect
import subprocess
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/archived-images'
#app.config['UPLOAD_FOLDER'] = 'static/KLING'
app.config['MASK_FOLDER'] = 'static/archived-masks'
app.config['STORE_FOLDER'] = 'static/archived-store'
#app.config['STORE_FOLDER'] = 'static/KLING'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

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
    image = Image.open(image_path).convert('L')
    threshold = 128
    image = image.point(lambda p: p > threshold and 255)
    mask_path = os.path.join(app.config['MASK_FOLDER'], 'binary_mask.png')
    image.save(mask_path)
    shutil.copy(mask_path, app.config['UPLOAD_FOLDER'])
    return mask_path

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





if __name__ == '__main__':
    app.run(debug=True)
