#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION

# **In this kernel, we will go through a mini project of Deep Learning where we will predict whether the cotton plants or their leaves are infected or fresh using ResNet50 transfer learning Model.**

# # 1) Import the libraries

# In[1]:


from tensorflow.keras.layers import Input,Dense,Flatten,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# # 2) Define the image size according to the standard set by ResNet50

# In[2]:


IMAGE_SIZE = [224,224]


# # 3) Define the training path and testing path of the images of cotton plants

# In[3]:


train_path = r'C:\Users\Vikash Chand\Downloads\CottonDiseaseDetection-main\CottonDiseaseDetection-main\Cotton Disease\train'
valid_path = r'C:\Users\Vikash Chand\Downloads\CottonDiseaseDetection-main\CottonDiseaseDetection-main\Cotton Disease\test'


# # 4) Load the pre-trained model and define the weights and input image size

# In[4]:


resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet',include_top=False)


# **weights='imagenet' means that we will use pre-trained weights of the imagenet which was used to train the resnet50 model.
# Since these pre-trained models were trained on imagenet database, it was classified for 1000 categories. But we have only 4 categories so we will remove the last layer and first layer i.e. include_top=False**

# # 5) Do not train the existing weights

# In[5]:


for layer in resnet.layers:    #layer.trainable=False means we dont want to retrain the existing weights.
    layer.trainable=False


# # 6) Use glob to get total categories so that we can add it at the bottom of our network

# In[7]:


folders=glob(r'C:\Users\Vikash Chand\Downloads\CottonDiseaseDetection-main\CottonDiseaseDetection-main\Cotton Disease\train\*')
folders


# In[23]:


resnet.output


# # 7) Now flatten the output

# In[24]:


x= Flatten()(resnet.output)


# # 8) Find the predictions and feed it to the model

# In[25]:


prediction = Dense(len(folders),activation='softmax')(x)


# # 9) Create model object

# In[26]:


model = Model(inputs=resnet.input, outputs=prediction)


# In[27]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# # 10) Use Image Data generator to import images from folder and for data augmentation

# In[28]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[30]:


train_datagen = ImageDataGenerator(rescale=1./255,shear_range = 0.2, zoom_range=0.2, horizontal_flip=True)


# In[31]:


test_datagen=ImageDataGenerator(rescale=1./255)


# In[32]:


training_set=train_datagen.flow_from_directory(r'C:\Users\Vikash Chand\Downloads\CottonDiseaseDetection-main\CottonDiseaseDetection-main\Cotton Disease\train',target_size=(224,224),batch_size=32,class_mode='categorical')


# In[33]:


test_set=test_datagen.flow_from_directory(r'C:\Users\Vikash Chand\Downloads\CottonDiseaseDetection-main\CottonDiseaseDetection-main\Cotton Disease\test',target_size=(224,224),batch_size=32,class_mode='categorical')


# In[34]:


r=model.fit(training_set, validation_data=test_set,epochs=5,steps_per_epoch=len(training_set),validation_steps=len(test_set))


# 
# **In this I have kept the epochs as 5 so the accuracy will be a bit less but if you increase the number of epochs, accuracy will definitely increase.**
# 

# 
