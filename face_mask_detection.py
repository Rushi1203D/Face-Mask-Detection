#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf 
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,Conv2D,Activation,Dropout
from keras import backend as k

from keras.models import Sequential,Model
from keras.models import load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import MaxPool2D,AveragePooling2D



# In[2]:


train_data = 'Train'
val_data ='Validation'
test_data = 'Test'


# In[3]:


os.listdir(train_data)


# In[4]:


train_datagen = ImageDataGenerator(rescale = 1/255.,
                                  zoom_range=0.15,
                                  width_shift_range = 0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.15)

val_datagen = ImageDataGenerator(rescale = 1/255.)

test_datagen = ImageDataGenerator(rescale = 1/255.)


# In[5]:


train_generator = train_datagen.flow_from_directory(train_data,target_size=(128, 128),batch_size=64,shuffle=True,class_mode='sparse')
val_generator = val_datagen.flow_from_directory(val_data,target_size=(128, 128),batch_size=64,shuffle=False,class_mode='sparse')
test_generator = test_datagen.flow_from_directory(test_data,target_size=(128, 128),batch_size=64,shuffle=False,class_mode='sparse')


# In[6]:


vgg19 = tf.keras.applications.vgg19.VGG19(
    include_top=False,
    weights='imagenet',
   input_shape=(128,128,3)
)


# In[7]:


vgg19.trainable = False


# In[8]:


model = keras.Sequential([
    vgg19,
    keras.layers.Flatten(),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=2, activation="softmax")
])


# In[9]:


model.summary()


# In[10]:


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[11]:


es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)


# In[12]:


mc = ModelCheckpoint('/content/gdrive/My Drive/best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)


# In[13]:


H = model.fit_generator(train_generator,
                        epochs=5,
                        verbose=1, 
                        validation_data=val_generator
                        )


# In[14]:


model.evaluate(test_generator)


# In[29]:


model.save('Face_mask_detection.h5')


# In[16]:


y_pred = model.predict(test_generator)


# In[17]:


y_true = test_generator.classes


# In[18]:


y_true


# In[19]:


y_pred


# In[20]:


y_pred = np.argmax(y_pred, axis=1)


# In[21]:


y_pred


# In[22]:


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[23]:


con_matrix = confusion_matrix(y_true, y_pred)


# In[24]:


print(con_matrix)


# In[25]:


disp = ConfusionMatrixDisplay(confusion_matrix=con_matrix,
                                 display_labels=test_generator.classes)


# In[26]:


disp.plot()


# In[27]:


report = classification_report(y_true, y_pred)


# In[28]:


print(report)


# In[ ]:




