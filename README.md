# Tree-species-classification-using-CNN


This script demonstrates how to use a Convolutional Neural Network (CNN) for image classification. The process includes several steps such as loading images, preprocessing, defining the CNN model, compiling and fitting the model, and calculating the accuracy. It's important to note that in this example, the dataset is only divided into training and testing sets, and only the test accuracy is printed in the results.

## Step 1: Loading Images

Load the images from the dataset. This step involves importing necessary libraries and then loading the dataset into memory.

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

```
# Step 2: Preprocessing
Preprocess the images by normalizing pixel values and performing any required transformations such as resizing or augmentation.

```python
## pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

## split dataset for training and validation
x_train,x_test,y_train,y_test = train_test_split(data, labels,
test_size=0.2)

## converting into categorical labels
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

```

# Step 3: Defining CNN Model
Define the architecture of the CNN model using Keras.

```python
#CNN model

model_cnn = models.Sequential([
        #cnn
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(250,250,3)),
        layers.MaxPool2D((2,2)),

        layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        
        layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        
        layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
        layers.MaxPool2D((2,2)),

        #Dense

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        #layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')             
])
```

# Step 4: Compiling the Model
Compile the model by specifying the loss function, optimizer, and evaluation metrics

```python
#Compiling model
model_cnn.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-4),
              metrics=['acc'])
```

# Step 5: Fitting the Model
Fitting the model on the training data. For the validation set use **"validation_split = 0.2"**

```python
#Model fitting
history=model_cnn.fit(x_train, y_train, epochs=30, batch_size=50, 
          #validation_split = 0.2,
          verbose=1,)
```

# Step 6: Calculating Accuracy
Finally, calculating the accuracy of the model.

```python
#Accuracy calculation
accuracy_score(Y_test, np.argmax(Y_pred, axis=1))
```

