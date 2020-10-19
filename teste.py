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

from PIL import Image

import cv2
import glob
import ntpath


def extract_shape_descriptor(image_path):
    try:
        file_name = ntpath.basename(image_path)
        print(file_name)
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
        splited_filename = file_name.split('.')
        new_file_name = splited_filename[0] + '_shape.' + splited_filename[1]
        print(new_file_name)
        # cv2.imwrite('./test-gun/nobackground.png', img_a*255)

        new_path = './test-gun/shape/' + new_file_name
        cv2.imwrite(new_path, masked)           # Save

        img = cv2.imread(new_path)
        img = cv2.medianBlur(img, 5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        cv2.imwrite('./test-gun/binary/'+new_file_name, th2)           # Save
        # plt.imshow(th2)
        # plt.show()
    except Exception as e:
        print(e)
        pass


# image_list = glob.glob('./test-gun/**/*.jpg', recursive=True)
image_list = glob.glob('./not-gun-dataset/**/*.jpg', recursive=True)
for image_path in image_list:
    extract_shape_descriptor(image_path)
