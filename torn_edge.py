from PIL import Image, ImageFilter, ImageDraw, ImageOps
import numpy as np
import logging
from sys import argv
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def create_torn_edge_mask(image_size, mask_path):
    width, height = image_size
    logging.info(f"Creating a torn edge mask for image size: {width}x{height}")

    # Parameters
    border_width = 50  # Transparent border width
    torn_width = 50    # Width of the torn effect

    # Create a mask with all white (255)
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)

    # Function to draw the torn edges
    def draw_torn_edge(start, edge_length, is_horizontal):
        for i in range(edge_length):
            if start == "top" and i >= border_width:
                offset = int(np.random.normal(0, torn_width))
                y = min(height - 1, border_width + offset)
                draw.line((i, border_width, i, y), fill=0)
            elif start == "bottom" and i >= border_width:
                offset = int(np.random.normal(0, torn_width))
                y = max(border_width, height - 1 - offset)
                draw.line((i, height - border_width - 1, i, y), fill=0)
            elif start == "left" and i >= border_width:
                offset = int(np.random.normal(0, torn_width))
                x = min(width - 1, border_width + offset)
                draw.line((border_width, i, x, i), fill=0)
            elif start == "right" and i >= border_width:
                offset = int(np.random.normal(0, torn_width))
                x = max(border_width, width - 1 - offset)
                draw.line((width - border_width - 1, i, x, i), fill=0)

    # Draw torn edges on all four sides
    logging.info("Drawing torn edges on all sides")
    draw_torn_edge("top", width, True)
    draw_torn_edge("bottom", width, True)
    draw_torn_edge("left", height, False)
    draw_torn_edge("right", height, False)

    # Apply Gaussian blur to smooth the edges
    logging.info("Applying Gaussian blur to the mask")
    mask = mask.filter(ImageFilter.GaussianBlur(5))  # Adjust blur effect if needed

    # Save the mask
    logging.info(f"Saving mask to {mask_path}")
    mask.save(mask_path, "PNG")

def apply_torn_edge_effect(image_path, mask_path, output_path):
    logging.info(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    logging.debug(f"Image size: {width}x{height}")

    logging.info(f"Loading mask: {mask_path}")
    mask = Image.open(mask_path).convert("L")

    # Apply the mask to the image
    logging.info("Applying mask to the image")
    result = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask)

    # Save the result image with torn effect
    logging.info(f"Saving image with torn edge effect: {output_path}")
    result.save(output_path, "PNG")

def crop_to_transparent_border(image_path, output_path, border_width):
    logging.info(f"Loading image for cropping: {image_path}")
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    logging.debug(f"Image size: {width}x{height}")

    # Create a new image with a transparent background and the original size
    final_image = Image.new("RGBA", (width + 2 * border_width, height + 2 * border_width), (255, 255, 255, 0))

    # Paste the original image onto the transparent canvas
    final_image.paste(image, (border_width, border_width), image)

    # Save the final image with a transparent border
    logging.info(f"Saving final image with transparent border: {output_path}")
    final_image.save(output_path, "PNG")

if __name__ == "__main__":
    if len(argv) != 3:
        logging.error("Usage: python script.py <input_image_path> <output_image_path>")
    else:
        # Paths for saving intermediate and final images
        base_path = 'static/archived-images/base.png'
        mask_path = 'static/archived-images/mask.png'
        final_image_path = argv[2]
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(final_image_path), exist_ok=True)

        # Create torn edge mask and save it
        image_path = argv[1]
        create_torn_edge_mask(Image.open(image_path).size, mask_path)
        
        # Apply torn edge effect and save intermediate result
        apply_torn_edge_effect(image_path, mask_path, base_path)
        
        # Crop to a transparent border and save the final image
        crop_to_transparent_border(base_path, final_image_path, 50)
