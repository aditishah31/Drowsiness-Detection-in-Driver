# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:17:55 2021

@author: aditi
"""
import sys
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
#import random
tf.__version__
print(tf. __version__)
print(sys.version)

tf.test.is_built_with_cuda()

tf.test.is_gpu_available()
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
#divide dataset in open eye and close eye folders.

raw_dataset_path = r'C:\Users\aditi\OneDrive\Desktop\UC\semester2\PRML\drowsieness_detction\mrlEyes_2018_01'

for dirpath, dirname, filenames in os.walk(raw_dataset_path):
    for i in  [file for file in filenames if  file.endswith('.png')]:
        if i.split('_')[4] == '0':
            shutil.copy(src = dirpath+'/'+i, dst = r'C:\Users\aditi\OneDrive\Desktop\UC\semester2\PRML\drowsieness_detction\dataset\close_eyes')
            
        else:
            shutil.copy(src = dirpath+'/'+i, dst = r'C:\Users\aditi\OneDrive\Desktop\UC\semester2\PRML\drowsieness_detction\dataset\open_eyes')
      
# # Creating Train / Val / Test folders

source_dir = r'C:\Users\aditi\OneDrive\Desktop\UC\semester2\PRML\drowsieness_detction\dataset' # data root path
classes_dir = ['\\close_eyes', '\\open_eyes'] #total labels

val_ratio = 0.15
test_ratio = 0.20

for cls in classes_dir:
    os.makedirs(source_dir +'\\train_dataset' + cls)
    os.makedirs(source_dir +'\\val_dataset' + cls)
    os.makedirs(source_dir +'\\test_dataset' + cls)


# Creating partitions of the data after shuffeling
    
for cls in classes_dir:
    src = source_dir + cls # Folder to copy images from


    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(allFileNames)* (1 - test_ratio))])


    train_FileNames = [src+'\\'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'\\' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'\\' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
#for cls in classes_dir:
    for name in train_FileNames:
        shutil.copy(name, source_dir +'\\train_dataset' + cls)

    for name in val_FileNames:
        shutil.copy(name, source_dir +'\\val_dataset' + cls)

    for name in test_FileNames:
        shutil.copy(name, source_dir +'\\test_dataset' + cls)
 
#  Data Preprocessing

# Preprocessing the Training set       
        
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_data = train_datagen.flow_from_directory(r'C:\Users\aditi\OneDrive\Desktop\UC\semester2\PRML\drowsieness_detction\dataset\new_train_dataset',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_data = test_datagen.flow_from_directory(r'C:\Users\aditi\OneDrive\Desktop\UC\semester2\PRML\drowsieness_detction\dataset\new_test_dataset',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

from keras.layers import Dropout,Input, Lambda, Dense, Flatten,MaxPooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# add preprocessing layer to the front of VGG
basemodel = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)

# don't train existing weights
for layer in basemodel.layers:
  layer.trainable = False
  
#outputlayers
  
headmodel = Flatten()(basemodel.output)
headmodel = Dense(64, activation='relu')(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(1,activation= 'sigmoid')(headmodel)

# create a model object
model = Model(inputs=basemodel.input, outputs=headmodel)

# view the structure of the model
model.summary()


model.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])

model.fit_generator(
  train_data,
  validation_data=test_data,
  epochs=5,
  steps_per_epoch=len(train_data),
  validation_steps=len(test_data)
)


##################SVM########################
#from keras.layers import Dropout,Input, Lambda, Dense, Flatten,MaxPooling2D
from keras.models import Model
#from keras.applications.mobilenet import MobileNet
#from keras.applications.mobilenet import preprocess_input


from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Flatten



# add preprocessing layer to the front of VGG
#basemodel = MobileNet(input_shape=(64,64,3), weights='imagenet', include_top=False)

# don't train existing weights
#for layer in basemodel.layers:
#  layer.trainable = False
# Initialising the CNN
cnn = Sequential()
# Step 1 - Convolution
cnn.add(Conv2D(filters=32,padding="same",kernel_size=3, activation='relu', strides=2, input_shape=[224, 224, 3]))

# Step 2 - Pooling
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(Conv2D(filters=32,padding='same',kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(Flatten())

# Step 4 - Full Connection
cnn.add(Dense(units=128, activation='relu'))

## For Binary Classification
cnn.add(Dense(1, kernel_regularizer=l2(0.01),activation
             ='linear'))

#headmodel = Dense(1,activation= 'sigmoid')(headmodel)

# create a model object
#model = Model(inputs=basemodel.input, outputs=headmodel)

# view the structure of the model
cnn.summary()

cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])
#model.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])

cnn.fit(
  train_data,
  validation_data=test_data,
  epochs=15
  #steps_per_epoch=len(train_data),
  #validation_steps=len(test_data)
  )

# save it as a h5 file


from tensorflow.keras.models import load_model

cnn.save('model_eye_detection.h5')
 
# load model
model = load_model('model_eye_detection.h5')

model.summary()


import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img(r'C:\Users\aditi\OneDrive\Desktop\s0001_00001_0_0_0_0_0_01.png', target_size = (224,224))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)


img_array = 