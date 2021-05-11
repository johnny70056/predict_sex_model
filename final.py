# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:05:41 2021

@author: johnn
"""
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os 
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

os.chdir('C:\\Users\\johnn\\OneDrive\\桌面\\AI專題\\圖像\\project1')

#%%
#呼叫模型
test = load_model('./InceptionResNetV2_sex_model2.h5')

#%%
import glob
x=[]
y=0
img_size = 128
picture_test =  os.listdir('./test_picture')
for i in picture_test: 
    img_file = glob.glob('./test_picture/'+i)

    for f in img_file:
        img = cv2.imread(f,1)
        img_resize = cv2.resize(img,(img_size,img_size))
        img_resize_reshape = (np.reshape(img_resize,(img_size,img_size,3))).astype(float)
        x.append(img_resize_reshape)
        x_data=np.array(x)

        new_output = test.predict(preprocess_input(x_data))
        if new_output[y][0] > new_output[y][1] :
            print('預測機率分別是:',new_output[y],'猜測是女生')
            y += 1
        else :
            print('預測機率分別是:',new_output[y],'猜測是男生')
            y += 1










