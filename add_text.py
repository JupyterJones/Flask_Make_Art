from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
import logging

app = Flask(__name__)

# Configurations
app.config['UPLOAD_FOLDER'] = 'static/archived-store/'
app.config['TEMP_FOLDER'] = 'static/temp/'
app.config['FONT_FOLDER'] = 'static/fonts/'
app.config['PUBLISH_FOLDER'] = 'static/archived-store/'

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Route for the form
@app.route('/', methods=['GET', 'POST'])
def add_text():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    
    if request.method == 'POST':
        image_file = request.form['image_file']
        text = request.form['text']
        position = (int(request.form['x_position']), int(request.form['y_position']))
        font_size = int(request.form['font_size'])
        color = request.form['color']
        font_path = os.path.join(app.config['FONT_FOLDER'], 'MerriweatherSans-Bold.ttf')
        font = ImageFont.truetype(font_path, font_size)

        logging.info(f"Processing image: {image_file}")
        logging.info(f"Text to add: '{text}' at position {position}, font size: {font_size}, color: {color}")

        # Open the image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
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
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
    image = Image.open(image_path)

    # Draw the final text on the image
    draw = ImageDraw.Draw(image)
    draw.text(position, final_text, font=font, fill=color)

    # Save the image with a unique UUID
    unique_filename = f"{uuid.uuid4()}.png"
    final_image_path = os.path.join(app.config['PUBLISH_FOLDER'], unique_filename)
    image.save(final_image_path)

    logging.info(f"Saved final image as: {unique_filename}")

    return redirect(url_for('add_text'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5100)
