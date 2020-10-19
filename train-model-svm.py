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



from keras.applications.resnet50 import ResNet50
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm



print(ELU)

def rotateImage(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols,rows))
    
    
def loadBlurImg(path, imgSize):
    img = cv2.imread(path)
    #angle = np.random.randint(-180, 180)
    #img = rotateImage(img, angle)
    img = cv2.blur(img,(5,5))
    img = cv2.resize(img, imgSize)
    return img

def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []
    
    for path in classPath:
        print(path)
        img = loadBlurImg(path, imgSize)        
        x.append(img)
        y.append(classLable)
        
    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = loadBlurImg(classPath[randIdx], imgSize)
        x.append(img)
        y.append(classLable)
        
    return x, y

def loadData(img_size, classSize):
    guns = glob.glob('./gun-dataset/**/*.jpg', recursive=True)
    
    imgSize = (img_size, img_size)
    xGun, yGun = loadImgClass(guns, 0, classSize, imgSize)
    print("There are", len(xGun), "gun images")
    
    X = np.array(xGun)
    y = np.array(yGun)
    return X, y


def buildNetwork(X, keepProb):
    mu = 0
    sigma = 0.3
    
    output_depth = {
        0 : 3,
        1 : 8,
        2 : 16,
        3 : 32,
        4 : 3200,
        5 : 240,
        6 : 120, 
        7 : 43,
    }
    
    #Layer 1: Convolutional + MaxPooling + ReLu + dropout. Input = 64x64x3. Output = 30x30x8.
    layer_1 = tf.Variable( tf.truncated_normal([5,5,output_depth[0],output_depth[1]], mean=mu, stddev=sigma))
    layer_1 = tf.nn.conv2d(X, filter=layer_1, strides=[1,1,1,1], padding ='VALID')
    layer_1 = tf.add(layer_1, tf.zeros(output_depth[1]))
    layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer_1 = tf.nn.dropout(layer_1, keepProb)
    layer_1 = tf.nn.relu(layer_1)
    
    return layer_1





def normalizeImages(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)
    
    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])
    
    images = images.reshape(images.shape) 
    return images

def preprocessData(images):
    return normalizeImages(images)


size = 128
classSize = 1000
scaled_X, y = loadData(size, classSize)

n_classes = len(np.unique(y))
print("Number of classes =", n_classes)

scaled_X = preprocessData(scaled_X)

label_binarizer = LabelBinarizer()

from keras.utils.np_utils import to_categorical
y = to_categorical(y)
print("y shape", y.shape)
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)


print("X_train shape", X_train.shape)


clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)
clf.fit(X_train)

test_hotdogs = glob.glob('./test-gun/**/*.jpg', recursive=True)
image_list = []
for image_in_folder in test_hotdogs:
    print(image_in_folder)
    img = cv2.imread(image_in_folder)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (128, 128))
    image_list.append(img)

images = np.array(image_list)
processedImages = preprocessData(images)

clf_prediction = clf.predict(processedImages)

print(clf_prediction)



# isolation_forest = IsolationForest(contamination=0.08, max_features=1.0,
#                          max_samples=1.0, n_estimators=40)  # Obtained using grid search
# isolation_forest.fit(X_train)
# isolation_forest_prediction = isolation_forest.predict(X_test)
