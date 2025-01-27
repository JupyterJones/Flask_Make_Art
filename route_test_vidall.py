from flask import Flask, render_template, request, jsonify
import os
import subprocess
import shutil
import time
import uuid
import datetime
import glob
from werkzeug.utils import secure_filename
from icecream import ic
app = Flask(__name__)

# Root directory for images and videos
IMAGE_ROOT = "static"
VIDEO_OUTPUT_DIR = "static/videos"

# Ensure the output directory exists
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

@app.route('/vidall', methods=['GET', 'POST'])
def vidall():
    """List all subdirectories containing images and view images dynamically."""
    # Function to check if a directory contains images
    def has_images(directory):
        valid_extensions = ('.png', '.jpg', '.jpeg')
        return any(
            f.endswith(valid_extensions) for f in os.listdir(directory)
        )

    # Filter directories that contain at least one image
    subdirectories = [
        d for d in os.listdir(IMAGE_ROOT)
        if os.path.isdir(os.path.join(IMAGE_ROOT, d)) and has_images(os.path.join(IMAGE_ROOT, d))
    ]
    ic(subdirectories)
    ic(len(subdirectories))
    # Handle POST request to show images from a selected directory
    selected_directory = None
    images = []
    if request.method == 'POST':
        selected_directory = request.form.get('directory')
        if selected_directory:

            directory_path = os.path.join(IMAGE_ROOT, selected_directory)
            if os.path.exists(directory_path):
                # List and sort files by modified time in reverse order
                images = sorted(
                    (f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg'))),
                    key=lambda x: os.path.getmtime(os.path.join(directory_path, x)),
                    reverse=True
                )
                images = [f"{selected_directory}/{img}" for img in images]

    return render_template(
        'vidall.html',
        subdirectories=subdirectories,
        selected_directory=selected_directory,
        images=images
    )


@app.route('/create_vidall_video', methods=['POST'])
def create_vidall_video():
    """Run the vidall command for the selected directory."""
    selected_directory = request.form.get('directory')
    ic(selected_directory)
    if not selected_directory:
        return jsonify({'error': 'No directory selected'}), 400

    input_path = os.path.join(IMAGE_ROOT, selected_directory)
    ic(input_path)
    if not os.path.exists(input_path):
        return jsonify({'error': f'Directory {input_path} does not exist'}), 400

    # Run the vidall command
    try:
        subprocess.run(
            ["./vidall.sh", input_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        video_path = os.path.join(VIDEO_OUTPUT_DIR, f"{selected_directory}.mp4")
        return jsonify({'success': True, 'video_path': video_path})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.stderr.decode()}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5400)
