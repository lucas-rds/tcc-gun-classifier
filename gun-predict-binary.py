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
import ntpath

from keras.utils.np_utils import to_categorical
from keras.models import load_model


def extract_shape_descriptor(image_path, tempc=1):
    #== Parameters =======================================================================
    BLUR = 25
    CANNY_THRESH_1 = 20
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (1.0, 1.0, 1.0)  # In BGR format

    #== Processing =======================================================================
    #-- Read image -----------------------------------------------------------------------
    img = cv2.imread(image_path)

    img = cv2.medianBlur(img, 5)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)
    # plt.imshow(gray)
    # plt.show()

    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(th2)
    # plt.show()

    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    # contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    im2, contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    #-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    #-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack = mask_stack.astype(
        'float32') / 255.0  # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)  # Blend
    # Convert back to 8-bit
    masked = (masked * 255).astype('uint8')

    # split image into channels
    c_red, c_green, c_blue = cv2.split(img)

    # merge with mask got on one of a previous steps
    img_a = cv2.merge(
        (c_red, c_green, c_blue, mask.astype('float32') / 255.0))

    # show on screen (optional in jupiter)
    # plt.imshow(img_a)
    # plt.show()

    # save to disk
    new_file_name = 'temp'+str(tempc)+'.jpg'
    print(new_file_name)
    # cv2.imwrite('./test-gun/nobackground.png', img_a*255)

    new_path = './temp/' + new_file_name
    cv2.imwrite(new_path, masked)           # Save

    img = cv2.imread(new_path)
    img = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    cv2.imwrite(new_path, th2)           # Save
    img = cv2.imread(new_path)
    return img

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

tempc = 1
for image_in_folder in sorted(test_images):
    print(image_in_folder)
    # img = cv2.imread(image_in_folder)
    img = extract_shape_descriptor(image_in_folder, tempc)
    # img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (128, 128))
    image_list.append(img)

    tempc += 1

images = np.array(image_list)


processedImages = preprocessData(images)


model = load_model('gun_model.h5')
predictions = model.predict_classes(processedImages,  verbose=1)
print(predictions)

