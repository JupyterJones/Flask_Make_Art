from flask import Flask, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFilter
import os

app = Flask(__name__)

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
    size = int(request.form.get('size', 50))
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

    # Save the result
    mask_path = f'static/archived_images/mask_{x}_{y}_{size}_{feather}_{aspect}.png'
    background.save(mask_path)

    return send_file(mask_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True,ho ,port=5000)
