import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import tensorflow as tf
from skimage import exposure
from tensorflow.contrib.layers import flatten
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.contrib.layers import flatten
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

import cv2
import glob

from keras.utils.np_utils import to_categorical

from keras.models import load_model


def toGray(images):
    images = 0.2989*images[:, :, :, 0] + 0.5870 * \
        images[:, :, :, 1] + 0.1140*images[:, :, :, 2]
    return images


def normalizeImages(images):
    images = (images / 255.).astype(np.float32)

    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])

    images = images.reshape(images.shape + (1,))
    return images


def preprocessData(images):
    grayImages = toGray(images)
    return normalizeImages(grayImages)


test_images = glob.glob('./test-data/**/*.jpg', recursive=True)
image_list = []
for image_in_folder in sorted(test_images):
    print(image_in_folder)
    img = cv2.imread(image_in_folder)
    # img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (128, 128))
    image_list.append(img)

images = np.array(image_list)


processedImages = preprocessData(images)


model = load_model('gun_model_v4.h5')
predictions = model.predict_classes(processedImages,  verbose=1)
print(predictions)

