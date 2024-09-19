import random
from PIL import Image, ImageFilter, ImageOps
import time
import cv2
from random import randint
import numpy as np
import os
#imgsize = (2000, 1000)
#seed_count = 10
#seed_max_size = 18000
count= 0
while count < 5:
    imgsize = (510, 766)
    seed_count = randint(6, 10)
    seed_max_size = randint(5000,16000)

    margin_h = 60
    margin_v = 60
    degradation = 10
    max_white = 100

    color = (0, 0, 0)
    img = Image.new("RGB", imgsize, "white")


    def next_points(point, avoid_points=[], shuffle=True):
        point_list = [p for p in 
                      [(point[0], point[1]+1), (point[0], point[1]-1), 
                       #(point[0]+1, point[1]+1), (point[0]-1, point[1]-1), 
                       #(point[0]-1, point[1]+1), (point[0]+1, point[1]-1), 
                       (point[0]+1, point[1]), (point[0]-1, point[1])]
                      if 0 < p[0] and 0 < p[1] < imgsize[1] and p not in avoid_points]

        for idx in range(len(point_list)):
            if point_list[idx][0] > imgsize[0]//2:
                point_list[idx] = (point[0], 
                                   point_list[idx][1] if point_list[idx][1] != point[1] else random.choice([point[1]+1,
                                                                                                            point[1]-1]))

        point_list = [p for p in point_list                  
                      if 0 < p[0] and 0 < p[1] < imgsize[1] and p not in avoid_points]

        if shuffle:
            random.shuffle(point_list)

        return point_list

    def degrade_color(color):
        return (color[0] + degradation, 
                color[1] + degradation,
                color[2] + degradation)

    def upgrade_color(color):
        return (color[0] - degradation//2, 
                color[1] - degradation//2,
                color[2] - degradation//2)

    def spread(img, point, color):
        if color[0] <= max_white and img.getpixel(point)[0] > color[0]:
            img.putpixel(point, color)
            points = next_points(point, shuffle=False)
            color = degrade_color(color)
            for point in points:
                spread(img, point, color)

    old_points = []
    posible_root_points = []
    for seed in range(0, seed_count):
        #print("Seed: %d" % seed)
        point = None
        while not point or point in old_points:
            point = (random.randrange(0 + margin_h, imgsize[0]//2), 
                     random.randrange(0 + margin_v, imgsize[1] - margin_v))
        old_points.append(point)
        posible_root_points.append(point)
        img.putpixel(point, color)

        seedsize = random.randrange(0, seed_max_size)
        #print("Seed size: %d" % seedsize)
        flow = 0
        for progress in range(0, seedsize):
            flow += 1
            points = next_points(point, old_points)
            try:
                point = points.pop()
            except IndexError:
                posible_root_points.remove(point)
                print("Looking for old points... Seed: %d Seed Size: %d "
                "Progress: %d Flow: %d Statistic: %d" % (seed,
                                                               seedsize,
                                                               progress,
                                                               flow, 
                                                               len(posible_root_points)))
                
                for idx in reversed(range(0, len(posible_root_points))):
                    points = next_points(posible_root_points[idx], old_points)
                    try:
                        point = points.pop()
                        #print("Using old point...")
                        flow = 0
                        break;
                    except IndexError:
                        posible_root_points.pop()
                if not point:
                    #print("No way!")
                    break

            old_points.append(point)
            posible_root_points.append(point)
            img.putpixel(point, color)

            for surr_point in points:
                spread(img, surr_point, degrade_color(color))

    print ("Cropping...")
    cropped = img.crop((0, 0, imgsize[0]//2, imgsize[1]))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.paste(cropped, (0, 0, imgsize[0]//2, imgsize[1]))
    img = img.filter(ImageFilter.GaussianBlur(radius=10))
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
    filename = time.strftime("blot/tmmmp.png")
    cv2.imwrite(filename, imagez)
    filename = time.strftime("blot/GOODblots%Y%m%d%H%M%S.png")
    ImageOps.expand(Image.open('blot/tmmmp.png').convert("L"),border=1,fill='white').save(filename)
    print ("GoodBlot : ",count)
    count=count+1    
