import pandas as pd
from pandas.core import indexing 
import numpy as np 
import csv
import os
import cv2
import math
from pathlib import Path

labels_path = './labels'
image_path = './images'

bbox_path ='./bbox'
os.makedirs(bbox_path, exist_ok=True)

label_files = os.listdir(labels_path)
image_files= os.listdir(image_path)

label_files.sort(reverse=False)
image_files.sort(reverse=False)

colors = [[255,0,0], 
[0,255,0], 
[0,0,255], 
[255,255,0], 
[255,0,255], 
[0,255,255], 
[255,255,255],
[230,230,230]]

counts = len(label_files)

for i in range(0, counts):

    temp1 = os.path.splitext(image_files[i])
    temp2 = os.path.splitext(label_files[i])
    
    if temp1[0]==temp2[0]:
        
        #label_file_size = os.path.getsize(labels_path+'/'+label_files[i])
        label_file_size=Path(labels_path+'/'+label_files[i]).stat().st_size
        #print(label_files[i], label_file_size)
        img = cv2.imread(image_path+'/'+image_files[i])
        height, width, channel = img.shape
        
        if label_file_size > 0 :
            #df = pd.read_csv(labels_path+'/'+label_files[i], sep = ' ') #, engine='python', encoding = "cp949")
            df = pd.read_table(labels_path+'/'+label_files[i], header=None, sep=" ")
    
            #print('============')
            #print(label_files[i], image_files[i], width, height)
            for index, row in df.iterrows():
                fx1 = float(df.loc[index, 1])
                fy1 = float(df.loc[index, 2])
                fx2 = float(df.loc[index, 3])
                fy2 = float(df.loc[index, 4])       
                if fx1==fy2==fx2==fy2 :
                    print(index+1, ' line ==============>    x1=y1=x2=y2 point', image_files[i], label_files[i])
                elif fx1==fx2 or fy1==fy2:
                    print(index+1, ' line ==============>    x1=x2 or y1=y2 line ', image_files[i], label_files[i])
                else:
                    x1 = round(fx1*width)
                    y1 = round(fy1*height)
                    x2 = round(fx2*width)
                    y2 = round(fy2*height)
                    #print(x1, y1, x2, y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), tuple(colors[int(df.loc[index,0])]), 2)
            #print('~~~~~~~~~~~~~')
            temp = os.path.splitext(image_files[i])
            temp_file = temp[0] + '_c.jpg'
            cv2.imwrite(bbox_path+'/'+temp_file, img)
        else:
            print('label file size = 0 bytes', label_files[i])
            temp = os.path.splitext(image_files[i])
            temp_file = '_'+ temp[0] + '.jpg'
            cv2.imwrite(bbox_path+'/'+temp_file, img)
    else:
        print('file name is not match ', image_files[i], label_files[i])
    
        