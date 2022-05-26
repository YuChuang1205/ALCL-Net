
#coding=gbk
'''
Created on 2020年3月7日

@author: yuchuang
'''
import os
import cv2
import numpy as np




#IOU
def cal_niou(image1,image2):
    
    image1 = np.where(image1>0.5,1,0)
    image2 = np.where(image2>0.5,1,0)
    n = image1*image2 
    u = image1+image2 - n
    
    n_sum = np.sum(n) + 1e-6
    u_sum = np.sum(u) +1e-6
    
    iou = n_sum/u_sum
    
    return iou



def cal_iou(image1,image2):
    image1 = np.where(image1>0.5,1,0)
    image2 = np.where(image2>0.5,1,0)
    n = image1*image2 
    u = image1+image2 - n
    
    n = np.sum(n)
    u = np.sum(u)
    return n,u
                

root_path = os.path.abspath('.')
pic_path = os.path.join(root_path,"data/test/mask")
pic_path2 = os.path.join(root_path,"data/test_results")
list1 = os.listdir(os.path.join(pic_path))
list1.sort(key= lambda x:int(x[:-4]))
list2 = os.listdir(os.path.join(pic_path2))
list2.sort(key= lambda x:int(x[:-4]))

niou_list = np.zeros(len(list1))
n_sum=0
u_sum=0
for i in range(len(list1)):
    print("-------------------------------------------------")
    print("对比的图片名：%s" %(list1[i]))
    print("测试的图片名：%s" %(list2[i]))
    image1 = cv2.imread(os.path.join(pic_path,list1[i]),cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(os.path.join(pic_path2,list2[i]),cv2.IMREAD_GRAYSCALE)
    #print(image2)
    
    image1 = image1/255
    image2 = image2/255
    
    niou = cal_niou(image1,image2)
    niou_list[i] = niou
    
    n,u = cal_iou(image1,image2)
    n_sum=n+n_sum
    u_sum=u+u_sum
    

    print("测试的图片为%s，其nIOU为：%f" %(list2[i],niou))

    print("-------------------------------------------------")

print(niou_list)


niou_mean = np.mean(niou_list)
iou_mean = n_sum/u_sum 

print("IoU为：%f"%(iou_mean))
print("nIoU为：%f" %(niou_mean))
