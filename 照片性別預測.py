# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:44:13 2021

@author: johnn


"""







#%%

import glob
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os 
import numpy as np
os.chdir('C:\\Users\\johnn\\OneDrive\\桌面\\AI專題\\圖像\\project1')
#%%

data = pd.read_csv('./wiki_crop/data_clean.csv')
data.head()

#%%
index = 18777

print(data.iloc[index,:])
path = os.path.join('./wiki_crop/data_clean', data.iloc[index,2])

#%%

img = cv2.imread(path)
plt.imshow(img[...,::-1])



#%%

#創建圖片年齡欄位

data['olds'] = data['photo_takens'] - data['year']
#%%
#確認資料完整性
data.info()
#%%
#暫時將性別為nan值的資料移除
data_no_nan = data.dropna()

#將年齡異常的資料暫時移除，模型訓練完之後再用來推測
a = data_no_nan['olds'] > 0
b = data_no_nan['olds'] < 110
data_clean = data_no_nan[( a & b )]
#%% 篩選尺寸大於256的圖片用做訓練

y = 0
select_data = []
for i in data_clean['path']:
    img_file = glob.glob('./wiki_crop/wiki_crop/'+i)
    for j in img_file:
        img = cv2.imread(j)
        a= img.shape
        if a[0] > 256 :
            select_data.append(i)
            y+=1
print(y)


data_for_train = data_clean[data_clean['path'].isin(select_data)]
#%%



data_resize = data_for_train[:5000]
#%%


x=[]
y=[]
img_size = 128
label = 0

#%%        

for i in data_resize['path']:
    img_file = glob.glob('./wiki_crop/wiki_crop/'+i)
    for f in img_file:
        img = cv2.imread(f,1) # cv2讀影像，0:GRAYSCALE 1:COLOR 2:UNCHANGED
        img_resize = cv2.resize(img,(img_size,img_size))
        img_resize_reshape = (np.reshape(img_resize,(img_size,img_size,3))).astype(float) # 將image做reshape成待會模型input的樣子
        x.append(img_resize_reshape) # 存放至x list當中
x_data=np.array(x) # 將讀完的資料轉成array形式    
#%%       
for a in data_resize['gender']:
    if a == 0:
            label = int(0)
    elif  a == 1:
            label = int(1)
    img_onehot = np.zeros(2,dtype=float) # :[0 0] →待會紀錄label (one hot型式)
    img_onehot[label] = 1
    y.append(img_onehot) # 將onehot完的label 存放至y list當中          
y_data=np.array(y) # 將紀錄完的label轉成array形式
#%%
#切割資料
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(x_data,y_data,test_size=0.2, random_state=42)

#%%

num_class = 2     #要分類出來的類別總數，男女=分2類
learning_rate = 0.001
loss = 'categorical_crossentropy'
metrics = 'accuracy'

#%%

from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

model_InceptionResNetV2 = InceptionResNetV2(weights='imagenet', include_top=False,input_shape=(img_size, img_size, 3))

#%%

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Dense, Dropout,Activation, Flatten, Conv2D,GlobalAveragePooling2D,MaxPooling2D)
                                    
x = GlobalAveragePooling2D()(model_InceptionResNetV2.output)
x = Dense(64, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)
new_model_InceptionResNetV2 = Model(inputs=model_InceptionResNetV2.input, outputs=predictions)

#%%

new_model_InceptionResNetV2.summary()

#%%

model_InceptionResNetV2.trainable=False
new_model_InceptionResNetV2.summary()

#%%

model_InceptionResNetV2.trainable = True

# 試試看不凍結任何層數直接fine tune all layers

# trainable_layer = 3
# for layer in model_InceptionResNetV2.layers[:-trainable_layer]:
#     layer.trainable = False

for layer in new_model_InceptionResNetV2.layers:
    print(layer, layer.trainable)



#%%
from tensorflow import keras
optimizer = keras.optimizers.Adam(lr = learning_rate)
new_model_InceptionResNetV2.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

#%%   Data Augmentation
batch_size = 32
num_steps = len(X_train) // batch_size 
num_epochs = 20

#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range=15,
                  width_shift_range=0.1,
                  height_shift_range=0.1,
                  horizontal_flip=True,
                  fill_mode='wrap',
                  preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#%%

train_generator = train_datagen.flow(X_train,y_train,batch_size=batch_size)

valid_generator = test_datagen.flow(X_valid,y_valid,batch_size=batch_size)


#%%

history = new_model_InceptionResNetV2.fit_generator(train_generator,
                              steps_per_epoch=num_steps,
                              epochs=num_epochs,
                              validation_data=valid_generator)


#%%  把模型存檔
!mkdir model-logs
new_model_InceptionResNetV2.save('./InceptionResNetV2_sex_model2.h5')

#%%    檢驗準確度
test = load_model('./InceptionResNetV2_sex_model.h5')
loss,acc = test.evaluate_generator(valid_generator, verbose=2)

#%%  混淆矩陣檢測


test_pred = new_model_InceptionResNetV2.predict(valid_generator)

#%%
test_pred_id = np.argmax(test_pred, axis=-1)
y_test_id = np.argmax(y_valid, axis=-1)

#%%
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true=y_test_id, y_pred=test_pred_id))

#%%

cnfm = confusion_matrix(y_true=y_test_id, y_pred=test_pred_id)
finall = pd.DataFrame(cnfm, columns=['Pred_male', 'Pred_female'], index=['Actual_male', 'Actual_female'])
print(finall)