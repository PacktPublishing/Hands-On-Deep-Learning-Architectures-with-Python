# MobileNEt

import keras
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.mobilenet import preprocess_input
from keras.models import Model


import numpy as np
import argparse
import matplotlib.pyplot as plt


model = keras.applications.mobilenet.MobileNet(weights = 'imagenet')

parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type = str, help = 'path to the image')
args = parser.parse_args()

# adding the path to image
IM_PATH = args.im_path

img = image.load_img(IM_PATH, target_size = (224, 224))
img = image.img_to_array(img)

img = np.expand_dims(img, axis = 0)
img = preprocess_input(img)
prediction = model.predict(img)

output = imagenet_utils.decode_predictions(prediction)

print(output)