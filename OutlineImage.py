"""
USE:
from OutlineImage import *
filename1 = '/home/jack/Desktop/Imagedata/0-original-images/07082orig.jpg' 
outfile_png = '/home/jack/Desktop/dockercommands/images/useresult.png'
outlineJ(filename1,outfile_jpg)
outlineP(filename1,outfile_png)

"""
from PIL import Image
import numpy as np
import cv2
import os
import imageio
from FileNameP import FilenameByTime
from pathlib import Path as change_ext
#p = change_ext('mysequence.jpg')
#p.rename(p.with_suffix('.png'))
try:
    os.makedirs("outlines")
except FileExistsError:
    # directory already exists
    pass
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

#image = cv2.imread('mahotastest/orig-color.png')
def change_extension(orig_file,new_extension):
    p = change_ext(orig_file)
    new_name = p.rename(p.with_suffix(new_extension))
    return new_name
    
def outlineJ(filename1,outfile_jpg,sigma=0.33):
    """
    USE:
    filename1 = '/home/jack/Desktop/Imagedata/0-original-images/07082orig.jpg' 
    outfile_jpg = '/home/jack/Desktop/dockercommands/images/useresult.png'
    outlineJ(filename1,outfile_jpg)
    """
    image = cv2.imread(filename1)
    edged = auto_canny(image, sigma=0.33)
    inverted = cv2.bitwise_not(edged)
    cv2.imwrite("outlines/temp2.png", inverted)
    cv2.imwrite(FilenameByTime("outlines/"), inverted)
    # Open Front Image
    #frontimage = Image.open('mahotastest/inverted-bitwise-note3_6.png').convert("1")
    frontimage = Image.open('outlines/temp2.png').convert("1")
    frontImage = frontimage.convert("RGBA")
    datas = frontImage.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    frontImage.putdata(newData)
    # Open Background Image
    background = Image.open(filename1)
    # Calculate width to be at the center
    width = (frontimage.width - frontimage.width) // 2
    # Calculate height to be at the center
    height = (frontimage.height - frontimage.height) // 2
    # Paste the frontImage at (width, height)
    background.paste(frontImage, (width, height), frontImage)
    # Save this image
    background.save(outfile_jpg)
    savefile = FilenameByTime("outlines/")
    background.save(savefile)
    #background = background.convert("RGB")
    return background
def outlineP(filename1,outfile_png):
    """
    USE:
    filename1 = '/home/jack/Desktop/Imagedata/0-original-images/07082orig.jpg' 
    outfile_jpg = '/home/jack/Desktop/dockercommands/images/useresult.png'
    outlineP(filename1,outfile_png)
    """    
    image = cv2.imread(filename1)
    edged = auto_canny(image, sigma=0.33)
    inverted = cv2.bitwise_not(edged)
    cv2.imwrite("outlines/temp2.png", inverted)
    cv2.imwrite(FilenameByTime("outlines/"), inverted)
    # Open Front Image
    #frontimage = Image.open('mahotastest/inverted-bitwise-note3_6.png').convert("1")
    frontimage = Image.open('outlines/temp2.png').convert("1")
    frontImage = frontimage.convert("RGBA")
    datas = frontImage.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    frontImage.putdata(newData)
    # Open Background Image
    background = Image.open(filename1)
    # Calculate width to be at the center
    width = (frontimage.width - frontimage.width) // 2
    # Calculate height to be at the center
    height = (frontimage.height - frontimage.height) // 2
    # Paste the frontImage at (width, height)
    background.paste(frontImage, (width, height), frontImage)
    # Save this image
    background.save(outfile_png, format="png")
    savefile = FilenameByTime("outlines/")
    background.save(savefile, format="png")
    #background = background.convert("RGB")
    return background
