#coding=gbk
'''
Created on 2020年3月27日

@author: yuchuang
'''

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.backend  as K
import keras
import os
import cv2

#判断目录是否存在，不存在则创建
def make_dir(path):
    if os.path.exists(path)==False:
        os.makedirs(path)

def soft_loss(y_true,y_pred):
    loss  = 1-(K.sum(y_true*y_pred)+1e-6)/(K.sum(y_true)+K.sum(y_pred)-K.sum(y_true*y_pred)+1e-6)
    return loss

def soft_acc(y_true, y_pred):
    acc_out = (K.sum(y_true*y_pred)+1e-6)/(K.sum(y_true)+K.sum(y_pred)-K.sum(y_true*y_pred)+1e-6)
    return acc_out


root_path = os.path.abspath('.')
pic_path = os.path.join(root_path,"data/test/image")
pic_out_path = os.path.join(root_path,"data/test_results")
make_dir(pic_out_path)

model_path = os.path.join(root_path,"seg_model_best.hdf5")
model = load_model(model_path,custom_objects={'soft_loss':soft_loss,'soft_acc':soft_acc})
#stride = 512
#image_size = 512

piclist = os.listdir(pic_path)
piclist.sort(key= lambda x:int(x[:-4])) 
for n in range(len(piclist)):
#     image = skimage.io.imread(os.path.join(pic_path,piclist[i]))
#     print(image.shape)
    new_name = piclist[n].split('.')[0]+'.png'
    image = cv2.imread(os.path.join(pic_path,piclist[n]))
    #print(image.shape)
    #print(image)
    image=image/255
    h,w,c = image.shape
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image,verbose=1)
    print(np.max(pred))
    pred = np.where(pred>=0.5,255,0)
    pred = pred.reshape((h,w,1))
    print(np.shape(pred))
    
    cv2.imwrite(os.path.join(pic_out_path,new_name), pred)
    print(np.max(pred),np.min(pred))
    #skimage.io.imsave(os.path.join(pic_out_path,new_name), image)
print("Done!!!")






























