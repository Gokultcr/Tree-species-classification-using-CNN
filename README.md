# Tree-species-classification-using-CNN


This Jupyter Notebook demonstrates how to use a Convolutional Neural Network (CNN) for image classification. The process includes several steps such as loading images, preprocessing, defining the CNN model, compiling and fitting the model, and calculating the accuracy. It's important to note that in this example, the dataset is only divided into training and testing sets, and only the test accuracy is printed in the results.

## Step 1: Loading Images

We first load the images from the dataset. This step involves importing necessary libraries and loading the dataset into memory.

```python


#Importing libraries

import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras import datasets, layers, models
import sklearn
from sklearn.metrics import accuracy_score
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, Activation, Flatten, Dropout
from keras import optimizers
from tensorflow.keras import optimizers
from keras.models import Sequential
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
