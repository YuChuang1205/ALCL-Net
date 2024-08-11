#coding=gbk
## Author: Yu Chuang
##
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


##ALCL-Net model
def ALCLNet(pretrained_weights = None,input_size = (512,512,3)):
    inputs = Input(input_size)
    conv1_1 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = ReLU()(conv1_1)
    
    
    conv1_2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = ReLU()(conv1_2)   
    
    

    
    #layer2_1
    conv2_1_1 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1_2)
    conv2_1_1 = BatchNormalization()(conv2_1_1)
    conv2_1_1 = ReLU()(conv2_1_1)
    
    conv2_1_2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_1_1)
    conv2_1_2 = BatchNormalization()(conv2_1_2)
    
    con_conv2_1 = Add()([conv1_2, conv2_1_2])
    con_conv2_1 = ReLU()(con_conv2_1)
    
    
    #layer2_2
    conv2_2_1 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv2_1)
    conv2_2_1 = BatchNormalization()(conv2_2_1)
    conv2_2_1 = ReLU()(conv2_2_1)
    
    conv2_2_2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_2_1)
    conv2_2_2 = BatchNormalization()(conv2_2_2)
    
    con_conv2_2 = Add()([con_conv2_1, conv2_2_2])
    con_conv2_2 = ReLU()(con_conv2_2)
    
    
    #layer2_3
    conv2_3_1 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv2_2)
    conv2_3_1 = BatchNormalization()(conv2_3_1)
    conv2_3_1 = ReLU()(conv2_3_1)
    
    
    conv2_3_2 = Conv2D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_3_1)
    conv2_3_2 = BatchNormalization()(conv2_3_2)
    
    con_conv2_3 = Add()([con_conv2_2, conv2_3_2])
    con_conv2_3 = ReLU()(con_conv2_3)            
    
    con_conv2_3_1 = Conv2D(32, 3, padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(con_conv2_3) 
    con_conv2_3_1 = BatchNormalization()(con_conv2_3_1)
    con_conv2_3_1 = ReLU()(con_conv2_3_1)
    
   
    #layer3_1
    conv3_1_1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)  
    conv3_1_1 = BatchNormalization()(conv3_1_1)
    conv3_1_1 = ReLU()(conv3_1_1)
    pooling3_1_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(conv3_1_1)    

    
    conv3_1_2 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(pooling3_1_1)
    conv3_1_2 = BatchNormalization()(conv3_1_2)
    
    con_conv3_1 = Add()([con_conv2_3_1,conv3_1_2])
    con_conv3_1 = ReLU()(con_conv3_1)
    
    #layer3_2
    conv3_2_1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv3_1)
    conv3_2_1 = BatchNormalization()(conv3_2_1)
    conv3_2_1 = ReLU()(conv3_2_1)
    
    conv3_2_2 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3_2_1)
    conv3_2_2  = BatchNormalization()(conv3_2_2)
    
    con_conv3_2 = Add()([con_conv3_1,conv3_2_2 ])
    con_conv3_2 = ReLU()(con_conv3_2)
    
    #layer3_3
    conv3_3_1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv3_2)
    conv3_3_1 = BatchNormalization()(conv3_3_1)
    conv3_3_1 = ReLU()(conv3_3_1)
    
    
    conv3_3_2 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3_3_1)
    conv3_3_2 = BatchNormalization()(conv3_3_2)
    
    con_conv3_3 = Add()([con_conv3_2,conv3_3_2])
    con_conv3_3 = ReLU()(con_conv3_3)    
    con_conv3_3_1 = Conv2D(64, 3, padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(con_conv3_3)  
    con_conv3_3_1 = BatchNormalization()(con_conv3_3_1)
    con_conv3_3_1 = ReLU()(con_conv3_3_1)
    
    
    
    #layer4_1
    conv4_1_1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
    conv4_1_1 = BatchNormalization()(conv4_1_1)
    conv4_1_1 = ReLU()(conv4_1_1)
    pooling4_1_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(conv4_1_1)  #120*160
    
    conv4_1_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pooling4_1_1)
    conv4_1_2 = BatchNormalization()(conv4_1_2)
    
    con_conv4_1 = Add()([con_conv3_3_1,conv4_1_2])
    con_conv4_1 = ReLU()(con_conv4_1)
    
    #layer4_2
    conv4_2_1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv4_1)
    conv4_2_1 = BatchNormalization()(conv4_2_1)
    conv4_2_1 = ReLU()(conv4_2_1)
    
    conv4_2_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4_2_1)
    conv4_2_2  = BatchNormalization()(conv4_2_2)
    
    con_conv4_2 = Add()([con_conv4_1,conv4_2_2 ])
    con_conv4_2 = ReLU()(con_conv4_2)
    
    
    
    #layer4_3
    conv4_3_1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv4_2)
    conv4_3_1 = BatchNormalization()(conv4_3_1)
    conv4_3_1 = ReLU()(conv4_3_1)
    
    
    conv4_3_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4_3_1)
    conv4_3_2 = BatchNormalization()(conv4_3_2)
    
    con_conv4_3 = Add()([con_conv4_2,conv4_3_2])      
    con_conv4_3 = ReLU()(con_conv4_3)                  
    con_conv4_3_1 = Conv2D(128, 3, padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(con_conv4_3)  #240*320
    con_conv4_3_1 = BatchNormalization()(con_conv4_3_1)
    con_conv4_3_1 = ReLU()(con_conv4_3_1)
     
     
     
     
    #layer5_1
    conv5_1_1 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
    conv5_1_1 = BatchNormalization()(conv5_1_1)
    conv5_1_1 = ReLU()(conv5_1_1)
    pooling5_1_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(conv5_1_1)
     
    conv5_1_2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pooling5_1_1)
    conv5_1_2 = BatchNormalization()(conv5_1_2)
     
    con_conv5_1 = Add()([con_conv4_3_1,conv5_1_2])
    con_conv5_1 = ReLU()(con_conv5_1)
     
    #layer5_2
    conv5_2_1 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv5_1)
    conv5_2_1 = BatchNormalization()(conv5_2_1)
    conv5_2_1 = ReLU()(conv5_2_1)
     
    conv5_2_2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5_2_1)
    conv5_2_2  = BatchNormalization()(conv5_2_2)
     
    con_conv5_2 = Add()([con_conv5_1,conv5_2_2 ])
    con_conv5_2 = ReLU()(con_conv5_2)
     
     
     
    #layer5_3
    conv5_3_1 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv5_2)
    conv5_3_1 = BatchNormalization()(conv5_3_1)
    conv5_3_1 = ReLU()(conv5_3_1)
     
     
    conv5_3_2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5_3_1)
    conv5_3_2 = BatchNormalization()(conv5_3_2)
     
    con_conv5_3 = Add()([con_conv5_2,conv5_3_2])      
    con_conv5_3 = ReLU()(con_conv5_3)
    con_conv5_3_1 = Conv2D(256, 3, padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(con_conv5_3) 
    con_conv5_3_1 = BatchNormalization()(con_conv5_3_1)
    con_conv5_3_1 = ReLU()(con_conv5_3_1)
     
     
     
     
    #layer6_1
    conv6_1_1 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv5_3)
    conv6_1_1 = BatchNormalization()(conv6_1_1)
    conv6_1_1 = ReLU()(conv6_1_1)
    pooling6_1_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(conv6_1_1)  #120*160
     
    conv6_1_2 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pooling6_1_1)
    conv6_1_2 = BatchNormalization()(conv6_1_2)
     
    con_conv6_1 = Add()([con_conv5_3_1,conv6_1_2])
    con_conv6_1 = ReLU()(con_conv6_1)
     
    #layer6_2
    conv6_2_1 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv6_1)
    conv6_2_1 = BatchNormalization()(conv6_2_1)
    conv6_2_1 = ReLU()(conv6_2_1)
     
    conv6_2_2 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6_2_1)
    conv6_2_2  = BatchNormalization()(conv6_2_2)
     
    con_conv6_2 = Add()([con_conv6_1,conv6_2_2 ])
    con_conv6_2 = ReLU()(con_conv6_2)
     
     
     
    #layer6_3
    conv6_3_1 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(con_conv6_2)
    conv6_3_1 = BatchNormalization()(conv6_3_1)
    conv6_3_1 = ReLU()(conv6_3_1)
     
     
    conv6_3_2 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6_3_1)
    conv6_3_2 = BatchNormalization()(conv6_3_2)
     
    con_conv6_3 = Add()([con_conv6_2,conv6_3_2])   
    con_conv6_3 = ReLU()(con_conv6_3)






    
    ## MLC6
    conv_mlc6_1 = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv6_3)
    con_mlc6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_mlc6_1)
#     conv_mlc6_1 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc6_1)
#     #print(conv_mlc5_1.get_shape)
#      
#      
#     conv_mlc6_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv6_3)
#     conv_mlc6_3 = Conv2D(256, 3, activation = 'relu', padding = 'same',dilation_rate=(3, 3), kernel_initializer = 'he_normal')(conv_mlc6_3)
#     conv_mlc6_3 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc6_3)
    
#     conv_mlc6_5 = Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv6_3)
#     conv_mlc6_5 = Conv2D(256, 3, activation = 'relu', padding = 'same',dilation_rate=(5, 5), kernel_initializer = 'he_normal')(conv_mlc6_5)
#     conv_mlc6_5 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc6_5)
    
#     conv_mlc6_7 = Conv2D(256, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv6_3)
#     conv_mlc6_7 = Conv2D(256, 3, activation = 'relu', padding = 'same',dilation_rate=(7, 7), kernel_initializer = 'he_normal')(conv_mlc6_7)
#     conv_mlc6_7 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc6_7)
    
#     conv_mlc6_9 = Conv2D(256, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv6_3)
#     conv_mlc6_9 = Conv2D(256, 3, activation = 'relu', padding = 'same',dilation_rate=(9, 9), kernel_initializer = 'he_normal')(conv_mlc6_9)
#     conv_mlc6_9 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc6_9)
    
#     conv_mlc6_11 = Conv2D(256, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv6_3)
#     conv_mlc6_11 = Conv2D(256, 3, activation = 'relu', padding = 'same',dilation_rate=(11, 11), kernel_initializer = 'he_normal')(conv_mlc6_11)
#     conv_mlc6_11 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc6_11)
    
# 
#    
#     con_mlc6 = Concatenate(axis=-1)([conv_mlc6_1,conv_mlc6_3])
#     
#     con_mlc6 = Lambda(lambda x: K.mean(x,-1))( con_mlc6)
    
   
    
    
    
    
    
    ## MLC5
    conv_mlc5_1 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv5_3)
    con_mlc5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_mlc5_1)
#     conv_mlc5_1 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc5_1)
#     #print(conv_mlc5_1.get_shape)
#      
#      
#     conv_mlc5_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv5_3)
#     conv_mlc5_3 = Conv2D(128, 3, activation = 'relu', padding = 'same',dilation_rate=(3, 3), kernel_initializer = 'he_normal')(conv_mlc5_3)
#     conv_mlc5_3 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc5_3)
#     
#     conv_mlc5_5 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv5_3)
#     conv_mlc5_5 = Conv2D(128, 3, activation = 'relu', padding = 'same',dilation_rate=(5, 5), kernel_initializer = 'he_normal')(conv_mlc5_5)
#     conv_mlc5_5 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc5_5)
    
#     conv_mlc5_7 = Conv2D(128, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv5_3)
#     conv_mlc5_7 = Conv2D(128, 3, activation = 'relu', padding = 'same',dilation_rate=(7, 7), kernel_initializer = 'he_normal')(conv_mlc5_7)
#     conv_mlc5_7 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc5_7)
    
#     conv_mlc5_9 = Conv2D(128, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv5_3)
#     conv_mlc5_9 = Conv2D(128, 3, activation = 'relu', padding = 'same',dilation_rate=(9, 9), kernel_initializer = 'he_normal')(conv_mlc5_9)
#     conv_mlc5_9 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc5_9)
    
#     conv_mlc5_11 = Conv2D(128, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv5_3)
#     conv_mlc5_11 = Conv2D(128, 3, activation = 'relu', padding = 'same',dilation_rate=(11, 11), kernel_initializer = 'he_normal')(conv_mlc5_11)
#     conv_mlc5_11 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc5_11)
#     
#     con_mlc5 = Concatenate(axis=-1)([conv_mlc5_1,conv_mlc5_3])
#     
#     con_mlc5 = Lambda(lambda x: K.mean(x,-1))( con_mlc5)
    
    
    
    ## MLC4  
    conv_mlc4_1 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
    con_mlc4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_mlc4_1)
#     conv_mlc4_1 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_1)
#     #print(conv_mlc5_1.get_shape)
#      
#      
#     conv_mlc4_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
#     conv_mlc4_3 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(3, 3), kernel_initializer = 'he_normal')(conv_mlc4_3)
#     conv_mlc4_3 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_3)
    
#     conv_mlc4_5 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
#     conv_mlc4_5 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(5, 5), kernel_initializer = 'he_normal')(conv_mlc4_5)
#     conv_mlc4_5 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_5)
    
#     conv_mlc4_7 = Conv2D(64, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
#     conv_mlc4_7 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(7, 7), kernel_initializer = 'he_normal')(conv_mlc4_7)
#     conv_mlc4_7 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_7)
    
#     conv_mlc4_9 = Conv2D(64, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
#     conv_mlc4_9 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(9, 9), kernel_initializer = 'he_normal')(conv_mlc4_9)
#     conv_mlc4_9 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_9)
    
#     conv_mlc4_11 = Conv2D(64, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv4_3)
#     conv_mlc4_11 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=(11, 11), kernel_initializer = 'he_normal')(conv_mlc4_11)
#     conv_mlc4_11 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc4_11)
    
#     con_mlc4 = Concatenate(axis=-1)([conv_mlc4_1,conv_mlc4_3])
#     
#     con_mlc4 = Lambda(lambda x: K.mean(x,-1))( con_mlc4)
    
    
    
    
    ## MLC3 
    conv_mlc3_1 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
    con_mlc3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_mlc3_1)
#     conv_mlc3_1 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_1)
#     #print(conv_mlc5_1.get_shape)
#      
#      
#     conv_mlc3_3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
#     conv_mlc3_3 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(3, 3), kernel_initializer = 'he_normal')(conv_mlc3_3)
#     conv_mlc3_3 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_3)
    
#     conv_mlc3_5 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
#     conv_mlc3_5 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(5, 5), kernel_initializer = 'he_normal')(conv_mlc3_5)
#     conv_mlc3_5 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_5)
    
#     conv_mlc3_7 = Conv2D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
#     conv_mlc3_7 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(7, 7), kernel_initializer = 'he_normal')(conv_mlc3_7)
#     conv_mlc3_7 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_7)
    
#     conv_mlc3_9 = Conv2D(32, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
#     conv_mlc3_9 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(9, 9), kernel_initializer = 'he_normal')(conv_mlc3_9)
#     conv_mlc3_9 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_9)
    
#     conv_mlc3_11 = Conv2D(32, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv3_3)
#     conv_mlc3_11 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=(11, 11), kernel_initializer = 'he_normal')(conv_mlc3_11)
#     conv_mlc3_11 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc3_11)
    
#     con_mlc3 = Concatenate(axis=-1)([conv_mlc3_1,conv_mlc3_3])
#     
#     con_mlc3 = Lambda(lambda x: K.mean(x,-1))(con_mlc3)
    
    
    
    
    
    ## MLC2 
    conv_mlc2_1 = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
    con_mlc2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_mlc2_1)
#     conv_mlc2_1 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_1)
#     #print(conv_mlc5_1.get_shape)
#      
#      
#     conv_mlc2_3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
#     conv_mlc2_3 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(3, 3), kernel_initializer = 'he_normal')(conv_mlc2_3)
#     conv_mlc2_3 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_3)
    
#     conv_mlc2_5 = Conv2D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
#     conv_mlc2_5 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(5, 5), kernel_initializer = 'he_normal')(conv_mlc2_5)
#     conv_mlc2_5 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_5)
    
#     conv_mlc2_7 = Conv2D(16, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
#     conv_mlc2_7 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(7, 7), kernel_initializer = 'he_normal')(conv_mlc2_7)
#     conv_mlc2_7 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_7)
    
#     conv_mlc2_9 = Conv2D(16, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
#     conv_mlc2_9 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(9, 9), kernel_initializer = 'he_normal')(conv_mlc2_9)
#     conv_mlc2_9 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_9)
    
#     conv_mlc2_11 = Conv2D(16, 11, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(con_conv2_3)
#     conv_mlc2_11 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=(11, 11), kernel_initializer = 'he_normal')(conv_mlc2_11)
#     conv_mlc2_11 = Lambda(lambda x: K.expand_dims(x,-1))(conv_mlc2_11)
    
#     con_mlc2 = Concatenate(axis=-1)([conv_mlc2_1,conv_mlc2_3])
#     
#     con_mlc2 = Lambda(lambda x: K.mean(x,-1))(con_mlc2)




    ##SBAM6
    up6 = UpSampling2D(size = (2,2),interpolation='bilinear')(con_mlc6)
    up6 = Conv2D(128, 1, padding = 'same', kernel_initializer = 'he_normal')(up6)
    up6 = BatchNormalization()(up6)
    up6 = ReLU()(up6)

    
    conv_blam5_x = Conv2D(128, 1, padding = 'same', kernel_initializer = 'he_normal')(con_mlc5)
    conv_blam5_x  = BatchNormalization()(conv_blam5_x)
    conv_blam5_x  = keras.layers.Activation('sigmoid')(conv_blam5_x)
    
    conv_blam5_y = Lambda(lambda x: x[0]*x[1])( [conv_blam5_x,up6])
    
    conv_blam5_out = Add()([con_mlc5,conv_blam5_y])
    
    
    
    ##SBAM5
    up5 = UpSampling2D(size = (2,2),interpolation='bilinear')(conv_blam5_out)
    up5 = Conv2D(64, 1, padding = 'same', kernel_initializer = 'he_normal')(up5)
    up5 = BatchNormalization()(up5)
    up5 = ReLU()(up5)

 
    conv_blam4_x = Conv2D(64, 1, padding = 'same', kernel_initializer = 'he_normal')(con_mlc4)
    conv_blam4_x  = BatchNormalization()(conv_blam4_x)
    conv_blam4_x  = keras.layers.Activation('sigmoid')(conv_blam4_x)
    
    conv_blam4_y = Lambda(lambda x: x[0]*x[1])( [conv_blam4_x,up5])
    
    conv_blam4_out = Add()([con_mlc4,conv_blam4_y])
    
    
    
    
    
    ##SBAM4
    up4 = UpSampling2D(size = (2,2),interpolation='bilinear')(conv_blam4_out)
    up4 = Conv2D(32, 1, padding = 'same', kernel_initializer = 'he_normal')(up4)
    up4 = BatchNormalization()(up4)
    up4 = ReLU()(up4)

    
    conv_blam3_x = Conv2D(32, 1, padding = 'same', kernel_initializer = 'he_normal')(con_mlc3)
    conv_blam3_x  = BatchNormalization()(conv_blam3_x)
    conv_blam3_x  = keras.layers.Activation('sigmoid')(conv_blam3_x)
    
    conv_blam3_y = Lambda(lambda x: x[0]*x[1])( [conv_blam3_x,up4])
    
    conv_blam3_out = Add()([con_mlc3,conv_blam3_y])
    
    
    ##SBAM3
    up3 = UpSampling2D(size = (2,2),interpolation='bilinear')(conv_blam3_out)
    up3 = Conv2D(16, 1, padding = 'same', kernel_initializer = 'he_normal')(up3)
    up3 = BatchNormalization()(up3)
    up3 = ReLU()(up3)
    
    
    conv_blam2_x = Conv2D(16, 1, padding = 'same', kernel_initializer = 'he_normal')(con_mlc2)
    conv_blam2_x  = BatchNormalization()(conv_blam2_x)
    conv_blam2_x  = keras.layers.Activation('sigmoid')(conv_blam2_x)
    
    conv_blam2_y = Lambda(lambda x: x[0]*x[1])( [conv_blam2_x,up3])
    
    conv_blam2_out = Add()([con_mlc2,conv_blam2_y])
    
    
    
    ##output
    conv_blam1_out = Conv2D(16, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_blam2_out)
    conv_blam1_out  = Conv2D(1, 1, activation = 'sigmoid')(conv_blam1_out)
    

    model = Model(input = inputs, output = conv_blam1_out)

    
#     model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

## ALCLNet()


