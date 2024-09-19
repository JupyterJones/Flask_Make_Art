#This one tries to to tell good blots from bad by particle_count`ing
# over twenty goes to one directory under twenty goes to another
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import random
from random import randint
import sys
import time
import cv2
import pylab
from scipy import ndimage
count = 0
while count < 200:
    imgsize = 640, 640
    color = (0, 0, 0)
    img = Image.new("RGB", imgsize, "white")
    max_range = randint(100,700)
    for j in range(0,max_range):
        start = (random.randrange(0, imgsize[0]/2), random.randrange(0, imgsize[1]))
        point = start
        img.putpixel(point, color)
        """
        point_list = [p for p in 
                      [(point[0], point[1]+1), (point[0], point[1]-1), 
                       #(point[0]+1, point[1]+1), (point[0]-1, point[1]-1), 
                       #(point[0]-1, point[1]+1), (point[0]+1, point[1]-1), 
                       (point[0]+1, point[1]), (point[0]-1, point[1])]
                      if 0 < p[0] and 0 < p[1] < imgsize[1] and p not in avoid_points]
        """

        blotsize = random.randrange(0, 640)
        for i in range(blotsize):
            directions = [(point[0], point[1]+1), (point[0], point[1]-1),
                         (point[0]+1, point[1]+1), (point[0]-1, point[1]-1), 
                         (point[0]-1, point[1]+1), (point[0]+1, point[1]-1),                       
                         (point[0]+1, point[1]), (point[0]-1, point[1])]
            toremove = []
            for direction in directions:
                if direction[0]>=(imgsize[0]/2) or direction[1]>=imgsize[1] or direction[0]<0 or direction[1]<0:
                    toremove.append(direction)
            for d in toremove:
                directions.remove(d)
            point = random.choice(directions)
            img.putpixel(point, color)


    cropped = img.crop((0, 0, imgsize[0]//2, imgsize[1]))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.paste(cropped, (0, 0, imgsize[0]//2, imgsize[1]))
    rad = randint(5,30)
    img = img.filter(ImageFilter.GaussianBlur(radius=rad))
    img.save("images/blot.png")


    def binarize_array(numpy_array, threshold=200):
        """Binarize a numpy array."""
        for i in range(len(numpy_array)):
            for j in range(len(numpy_array[0])):
                if numpy_array[i][j] > threshold:
                    numpy_array[i][j] = 255
                else:
                    numpy_array[i][j] = 0
        return numpy_array

    filename0=('images/blot.png')

    im = Image.open(filename0)
    im_grey = im.convert('LA') # convert to grayscale
    width,height = im.size

    total=0
    for i in range(0,width):
        for j in range(0,height):
            total += im_grey.getpixel((i,j))[0]

    mean = total / (width * height)

    image_file = Image.open(filename0)
    imagex = image_file.convert('L')  # convert image to monochrome
    imagey = np.array(imagex)
    #imagez = binarize_array(imagey, threshold)
    imagez = binarize_array(imagey, mean)
    time.sleep(2)
    filename = time.strftime("TEMP/tmpp.png")
    cv2.imwrite(filename, imagez)

    im = cv2.imread('TEMP/tmpp.png')
    pylab.figure(0)
    pylab.imshow(im)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    maxValue = 255
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C#cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresholdType = cv2.THRESH_BINARY#cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
    blockSize = 5 #odd number like 3,5,7,9,11
    C = -3 # constant to be subtracted
    im_thresholded = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C) 
    labelarray, particle_count = ndimage.measurements.label(im_thresholded)
    
    if particle_count < 20:
        print"JUST RIGHT : ",particle_count
        filename = time.strftime("TEMP/good%Y%m%d%H%M%S.png")
        ImageOps.expand(Image.open('TEMP/tmpp.png').convert("RGB"),border=30,fill='red').save(filename)
        #cv2.imwrite(filename, imagez)
    else:
        print "TOO MUCH : ",particle_count
        filename = time.strftime("junk/toomany%Y%m%d%H%M%S.png")
        ImageOps.expand(Image.open('TEMP/tmpp.png').convert("RGB"),border=30,fill='red').save(filename)
        #cv2.imwrite(filename, imagez)
    print count    
    count = count+1


