''' want to use a Flask app to upload a file to a specific upload directory.
ChatGPT
To create a Flask app that allows you to upload a file to a specific directory, you can follow these steps. I'll provide a complete script with detailed comments and logit.
Install Flask: Ensure you have Flask installed. You can install it using pip if you haven't already:
pip install Flask
Create the Flask App: Below is the complete script for the Flask application:
'''
import os
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
import datetime
import inspect


# Configuration
UPLOAD_FOLDER = 'static/archived-images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Logit function to log messages with timestamp, filename
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
        with open("exp_log.txt", "a") as file:
            # Write the log message to the file
            file.write(log_message)

            # Print the log message to the console
            print(log_message)

    except Exception as e:
        # If an exception occurs during logit, print an error message
        print(f"Error occurred while logit: {e}")

logit("App started")
# Check if the extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index_upload():
    logit("Rendering upload form.")
    return render_template('upload.html')

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
    return f"File {filename} uploaded successfully!"

if __name__ == "__main__":
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        logit(f"Created upload directory at {UPLOAD_FOLDER}.")
    
    app.secret_key = 'supersecretkey'
    app.run(debug=True, host='0.0.0.0', port=5100)
'''
Create an HTML Upload Form: Save this as upload.html in a templates directory in the same folder as your Flask app.
html
Copy code
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Upload new File</title>
</head>
<body>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data action="{{ url_for('upload_file') }}">
        <input type=file name=file>
        <input type=submit value=Upload>
    </form>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul>
        {% for message in messages %}
          <li>{{ message }}</li>
        {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}
</body>
</html>
Run the Flask App:
sh
Copy code
python app.py
Replace /path/to/your/upload/directory with the path where you want to save the uploaded files. This script includes logit, which will help you debug and trace the flow of the application. The logit statements provide insights into the different stages of file handling and any potential issues that may arise.
'''