#!/home/jack/miniconda3/envs/cloned_base/bin/python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from random import randint
from PIL import Image, ImageDraw, ImageFont, ImageChops 
import os
import sys
import markovify
import twython
from twython import Twython
import time
#nap=randint(10,400)
#time.sleep(nap)
path = r"/home/jack/Desktop/Flask_Make_Art/static/archived_resource/"
base_image = random.choice([
    x for x in os.listdir(path)
    if os.path.isfile(os.path.join(path, x))
])
filename0=(path+base_image)

#base_image ="/home/jack/Desktop/deep-dream-generator/notebooks/new/2/"
square_size = 10
offset_factor = 20
darken_factor = 0.1
image = Image.open(filename0)
size = image.size

# create numpy array from the opened image 
im = np.array(image, dtype=np.uint8)

# Plot figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Add random offset to tuple based on offset_factor
def add_offet(t):
    t[0] = random.randint(t[0], (t[0] + offset_factor))
    t[1] = random.randint(t[1], (t[1] + offset_factor))

    return t


# Create a Rectangle patch
for w in range(0, size[0], square_size):
    for h in range(0, size[1], square_size):
        #print str(w) + ':' + str(h)
        #square_size = randint(10,35)
        #offset_factor = randint(10,35)
    
        # Get the average color of the section
        rect = im[h:h+square_size, w:w+square_size]
        mean = rect.mean(axis=(0,1))

        # Convert to hex value
        face_color = '#%02x%02x%02x' % (int(mean[0]), int(mean[1]), int(mean[2]))
        edge_color = '#%02x%02x%02x' % (int(mean[0] - (mean[0] * darken_factor)), int(mean[1]  - (mean[1] * darken_factor)), int(mean[2] - (mean[2] * darken_factor)))

        # Dont draw outline with the dominant color
        z_order = 2
        if '#e' in face_color or '#f' in face_color:
            line_width = 0.0
        else:
            line_width = 0.1
            z_order = 3

        points = [add_offet([w, h]), add_offet([w + square_size, h]), add_offet([w, h + square_size])]
        triangle1 = patches.Polygon(points, edgecolor=edge_color, linewidth=line_width, facecolor=face_color, zorder=z_order)

        # Second triangle
        points2 = [add_offet([w, h + square_size]), add_offet([w + square_size, h + square_size]), add_offet([w + square_size, h])]
        triangle2 = patches.Polygon(points2, edgecolor=edge_color, linewidth=line_width, facecolor=face_color, zorder=z_order)

        # Square in background
        rec = patches.Rectangle((w,h),square_size,square_size,linewidth=0.0, edgecolor=edge_color, facecolor=face_color, zorder=1)

        # Add the patch to the Axes
        ax.add_patch(triangle1)
        ax.add_patch(triangle2)
        ax.add_patch(rec)

plt.axis('off')
plt.savefig("images/Sranger-Tri-001a.jpg", bbox_inches='tight', dpi=200)
img = Image.open("images/Sranger-Tri-001a.jpg")
nonwhite_positions = [(x,y) for x in range(img.size[0]) for y in range(img.size[1]) if img.getdata()[x+y*img.size[0]] != (255,255,255)]
rect = (min([x for x,y in nonwhite_positions]), min([y for x,y in nonwhite_positions]), max([x for x,y in nonwhite_positions]), max([y for x,y in nonwhite_positions]))
img.crop(rect).save('images/Sranger-Tri-001-crop2a.jpg')
#Create a second image
square_size = 10
offset_factor = 20
darken_factor = 0.1
image = Image.open(filename0)
size = image.size

# Create image np array
im = np.array(image, dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Add random offset to tuple based on offset_factor
def add_offet(t):
    t[0] = random.randint(t[0], (t[0] + offset_factor))
    t[1] = random.randint(t[1], (t[1] + offset_factor))

    return t


def generate_the_word(infile):
        with open(infile) as f:
            contents_of_file = f.read()
        lines = contents_of_file.splitlines()
        line_number = random.randrange(0, len(lines))
        return lines[line_number]
        #textin = (generate_the_word("wordcloud.txt"))

# Create a Rectangle patch
for w in range(0, size[0], square_size):
    for h in range(0, size[1], square_size):
        #print str(w) + ':' + str(h)
        #square_size = randint(10,35)
        #offset_factor = randint(10,35)
    
        # Get the average color of the section
        rect = im[h:h+square_size, w:w+square_size]
        mean = rect.mean(axis=(0,1))

        # Convert to hex value
        face_color = '#%02x%02x%02x' % (int(mean[0]), int(mean[1]), int(mean[2]))
        edge_color = '#%02x%02x%02x' % (int(mean[0] - (mean[0] * darken_factor)), int(mean[1]  - (mean[1] * darken_factor)), int(mean[2] - (mean[2] * darken_factor)))

        # Dont draw outline with the dominant color
        z_order = 2
        if '#e' in face_color or '#f' in face_color:
            line_width = 0.0
        else:
            line_width = 0.1
            z_order = 3

        points = [add_offet([w, h]), add_offet([w + square_size, h]), add_offet([w, h + square_size])]
        triangle1 = patches.Polygon(points, edgecolor=edge_color, linewidth=line_width, facecolor=face_color, zorder=z_order)

        # Second triangle
        points2 = [add_offet([w, h + square_size]), add_offet([w + square_size, h + square_size]), add_offet([w + square_size, h])]
        triangle2 = patches.Polygon(points2, edgecolor=edge_color, linewidth=line_width, facecolor=face_color, zorder=z_order)

        # Square in background
        rec = patches.Rectangle((w,h),square_size,square_size,linewidth=0.0, edgecolor=edge_color, facecolor=face_color, zorder=1)

        # Add the patch to the Axes
        ax.add_patch(triangle1)
        ax.add_patch(triangle2)
        ax.add_patch(rec)

plt.axis('off')
plt.savefig("images/Sranger-Tri-001a.jpg", bbox_inches='tight', dpi=200)
img = Image.open("images/Sranger-Tri-001a.jpg")
nonwhite_positions = [(x,y) for x in range(img.size[0]) for y in range(img.size[1]) if img.getdata()[x+y*img.size[0]] != (255,255,255)]
rect = (min([x for x,y in nonwhite_positions]), min([y for x,y in nonwhite_positions]), max([x for x,y in nonwhite_positions]), max([y for x,y in nonwhite_positions]))
img.crop(rect).save('images/Sranger-Tri-001-crop2b.jpg')


img0 = Image.open("images/Sranger-Tri-001-crop2a.jpg")
img1 = Image.open("images/Sranger-Tri-001-crop2b.jpg")
blen = ImageChops.blend(img0, img1, .9)
blen.save('images/Sranger-Tri-001-crop2b.jpg')

#---------------


base = Image.open('images/Sranger-Tri-001-crop2b.jpg').convert('RGBA')
 
#8 5 4 6 3 2
# make a blank image for the text, initialized to transparent text color
txt = Image.new('RGBA', base.size, (255,255,255,0))
def generate_the_word(infile):
    with open(infile) as f:
            contents_of_file = f.read()
    lines = contents_of_file.splitlines()
    line_number = random.randrange(0, len(lines))
    return lines[line_number]
    

#base = Image.open('images/NewFolder/lightning01.jpg').convert('RGBA')
#8 5 4 6 3 2
# make a blank image for the text, initialized to transparent text color
txt = Image.new('RGBA', base.size, (255,255,255,0))

# get a font
fnt = ImageFont.truetype("/home/jack/fonts/Calligraffitti-Regular.ttf", 40)
# get a drawing context
d = ImageDraw.Draw(txt)

width, height = base.size
# calculate the x,y coordinates of the text
#marginx = 325
#marginy = 75
marginx = 25
marginy = 50
x = width - marginx
y = height - marginy
signature_ = "The TwitterBot Project" 
d.text((600,580), signature_, font=fnt, fill=(0,0,0,256))
d.text((599,578), signature_, font=fnt, fill=(256,256,256,256))
out = Image.alpha_composite(base, txt)
out.save("images/tmp.png", "png")
# save the image then reopen to put a title
base = Image.open('images/tmp.png').convert('RGBA')
#8 5 4 6 3 2
# make a blank image for the text, initialized to transparent text color
txt = Image.new('RGBA', base.size, (255,255,255,0))

# get a font
fnt = ImageFont.truetype("/home/jack/fonts/Calligraffitti-Regular.ttf", 40)
# get a drawing context
d = ImageDraw.Draw(txt)

width, height = base.size
# calculate the x,y coordinates of the text
#marginx = 325
#marginy = 75
x = 90
y = height+20
#generate a title
title = (generate_the_word("titles.txt"))
d.text((35,40), title , font=fnt, fill=(256,256,256,256))
out2 = Image.alpha_composite(base, txt)
out2.save("images/TM_POST.png", "png")

#API Key UR9vZ3paf4R1XROQSxa1E9yuK
#API Key Secret bG0BtkdPBY6XNBrEQ2pd5hNo74hJwXoqXJdZLos0ofiUZwodzQ
#Access Token  296906916-VaaBXCF0rfuq9CdcWncOfc2fTmZkM4Gz1mwOICbL
#Access Token  9ds385rKu1EjNYKdgvyH3fb0P6ioKicye6TuInwnhPWVv

#removed keys for privacy reasons
CONSUMER_KEY = 'UR9vZ3paf4R1XROQSxa1E9yuK'
CONSUMER_SECRET = 'bG0BtkdPBY6XNBrEQ2pd5hNo74hJwXoqXJdZLos0ofiUZwodzQ'
ACCESS_KEY = '296906916-VaaBXCF0rfuq9CdcWncOfc2fTmZkM4Gz1mwOICbL'
ACCESS_SECRET = '9ds385rKu1EjNYKdgvyH3fb0P6ioKicye6TuInwnhPWVv'

twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_KEY, ACCESS_SECRET)
#path = 'images/NewFolder'
with open("sart.txt") as f:
    data = f.read()
data_model = markovify.Text(data)
STR = data_model.make_sentence()
#STR = ("#All_in_One - #WordCloud #Create - Added ability to randomly choose an image background  #Automated")
#PATH = "/home/jack/Desktop/deep-dream-generator/notebooks/STUFF/experiment/experiment8.jpg"
PATH = "images/TM_POST.png"
#PATH = "result.png"
# 1 , 2, 3, 12, 5, 15, 8, 6
#photo = open('/home/jack/Desktop/deep-dream-generator/notebooks/images/'+file_list[rnd]+'.jpg','rb')

photo = open(PATH,'rb')
response = twitter.upload_media(media=photo)
twitter.update_status(status=STR, media_ids=[response['media_id']])
