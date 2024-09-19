from PIL import Image
import PIL
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import time
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from PIL import Image
import random
import os
from sys import argv

directory ='preped_images'
if not os.path.exists(directory):
    os.makedirs(directory)
directory ='_images'
if not os.path.exists(directory):
    os.makedirs(directory)   
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)
    
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img    
def prep_content(content):
    PI = Image.open(content)
    PI = PI.resize((512,768), Image.NEAREST)
    CFilename = os.path.basename(content)
    PI.save("preped_images/Content_"+CFilename)
    Content_data="preped_images/Content_"+CFilename
    return Content_data
    
def prep_style(style):
    PI = Image.open(style)
    PI = PI.resize((512,768), Image.NEAREST)
    SFilename = os.path.basename(style)
    PI.save("preped_images/Style_"+SFilename)
    Style_data="preped_images/Style_"+SFilename
    return Style_data    


#path = r"/home/jack/Desktop/collections/images/Alien_Comic/"
#path = r"/home/jack/Desktop/NEW_IMAGES/june20/"
#path = r'/home/jack/Desktop/NEW_IMAGES/screen_caps/'
#path = r'/home/jack/Desktop/NEW_IMAGES/images/'
#path = r'/home/jack/Desktop/collections/images/apocalyptic_park/'
#path = r'/home/jack/Desktop/collections/images/z-civat/'
#/home/jack/Desktop/collections/images/yodayo_ant_Aliens/
path = r'/home/jack/Desktop/collections/images/yodayo_ant_Aliens/'
#/home/jack/Desktop/collections/images/alien_night_life_2/
path = r'/home/jack/Desktop/collections/images/alien_night_life_2/'
#/home/jack/Desktop/collections/images/may14/
path = r'/home/jack/Desktop/collections/images/may14/'
#/home/jack/Desktop/collections/images/june17/
path = r'/home/jack/Desktop/collections/images/june17/'
#/home/jack/Desktop/collections/images/tensor_art/
#path = r'/home/jack/Desktop/collections/images/tensor_art/'
#/home/jack/Desktop/collections/images/yodayo_new/
path = r'/home/jack/Desktop/collections/images/yodayo_new/'
#/home/jack/Desktop/collections/images/image_art/
path = r'/home/jack/Desktop/collections/images/image_art/'
#/home/jack/Desktop/NEW_IMAGES/static/images/
path = r'/home/jack/Desktop/NEW_IMAGES/static/images/'
#/home/jack/Desktop/HDD500/collections/images/Aztec_girl/
path = r'/home/jack/Desktop/HDD500/collections/images/Aztec_girl/'
#/home/jack/Desktop/HDD500/collections/images/beautiful_feb19/
path = r'/home/jack/Desktop/HDD500/collections/images/beautiful_feb19/'
#/home/jack/Desktop/HDD500/collections/images/z-yoday_feb10/
path = r'/home/jack/Desktop/HDD500/collections/images/z-yoday_feb10/'
#/home/jack/Desktop/HDD500/collections/images/z-yday-cheese/
path = r'/home/jack/Desktop/HDD500/collections/images/z-yday-cheese/'
#/home/jack/Desktop/HDD500/collections/images/z-leo-z/
path = r'/home/jack/Desktop/HDD500/collections/images/z-leo-z/'



#path = argv[1]
base_image = random.choice([
    x for x in os.listdir(path)
    if os.path.isfile(os.path.join(path, x)) and (x.endswith('.jpg') or x.endswith('.png'))
])
content=(path+base_image)
con=Image.open(content)
CFilename = os.path.basename(content)
con.save("_images/Content_"+CFilename)
print("content"+path+base_image)


#path2 = r'/home/jack/Desktop/collections/images/accident3/'
#path2 = r'/mnt/HDD500/collections/images/dec3_1/styles/jpgs/'
path2 = r'/mnt/HDD500/fast-neural-style/images/styles/'
#/home/jack/Desktop/collections/images/may14/
#path2 = r'/home/jack/Desktop/collections/images/may14/'
#/home/jack/Desktop/collections/images/tensor_art/
#path2 = r'/home/jack/Desktop/collections/images/tensor_art/'
#/home/jack/Desktop/collections/images/yodayo_new/
#path2 = r'/home/jack/Desktop/collections/images/yodayo_new/'
#/home/jack/Desktop/collections/images/yodayo_ant_Aliens/
#path2 = r'/home/jack/Desktop/collections/images/yodayo_ant_Aliens/'
#/home/jack/Desktop/HDD500/collections/images/z-yodayofeb6/
#path2 = r'/home/jack/Desktop/HDD500/collections/images/z-yodayofeb6/'
#/home/jack/Desktop/HDD500/collections/images/z-yoday_feb10/
#path2 = r'/home/jack/Desktop/HDD500/collections/images/z-yoday_feb10/'

#path2 = argv[2]
base_image = random.choice([
    x for x in os.listdir(path2)
    if os.path.isfile(os.path.join(path2, x)) and (x.endswith('.jpg') or x.endswith('.png'))
])
style=(path2+base_image)
sty=Image.open(style)
SFilename = os.path.basename(style)
sty.save("_images/Style_"+SFilename)
print("style"+path2+base_image)
content_image = load_img(prep_content(content))
style_image = load_img(prep_style(style))
print(content_image.size)
print(style_image.size)

hub_model = hub.load("http://0.0.0.0:8000/Models/magenta_arbitrary-image-stylization-v1-256_2.tar.gz")
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
im = tensor_to_image(stylized_image)
timestr = time.strftime("%Y%m%d-%H%M%S")
savefile = "_images/Result"+timestr+".jpg"
im.save(savefile)
print(im.size)
print("Saved to: "+savefile)
