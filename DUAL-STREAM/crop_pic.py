import os
import pandas as pd
import numpy as np
import dlib
import time
from PIL import Image
import cv2
from dlib68 import getxy
import scipy.io
from limited import limits
from apexs import apexs,crop_face,imflow

def crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path,afflow_pic_path):
    sizea = 360
#    onset_crop,apex_crop,sizea,onx,ony,apx,apy = crop_face(onset_path,apex_path,sizea)
    onset_crop,apex_crop = crop_face(onset_path,apex_path,sizea)
    size = 360
#    xl,xr,yl,yr = limits(onx,ony,size,sizea)
#    onset_crop = onset_crop[yl:yr,xl:xr]
#    xl,xr,yl,yr = limits(apx,apy,size,sizea)
#    apex_crop = apex_crop[yl:yr,xl:xr]
#    onset_crop = cv2.resize(onset_crop,(size,size))
#    apex_crop = cv2.resize(apex_crop,(size,size))    
    onset_cropped = Image.fromarray(cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB))
    apex_cropped = Image.fromarray(cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB))
    onset_cropped.save(onset_save_path)
    apex_cropped.save(apex_save_path)
    onsets = cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB)
    apexs = cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB)
    x,y =  getxy(onsets,size)
    if x!=[] and y!=[]:
        xydata={'x':x,'y':y}
        xydatas = pd.DataFrame(data=xydata)
        xydatas.to_csv('xydatas.csv')
    if x==[] or y ==[]:
        x,y =  getxy(apexs,size)
    if x==[] or y ==[]:
        xy = pd.read_csv('xydatas.csv', header=0)
        x = xy['x']
        y = xy['y']       
    x,y = np.array(x),np.array(y)
    
#    path_xy = onset_save_path+'xy.mat'
#    scipy.io.savemat(path_xy,{'x':x, 'y':y})
    af = imflow(x,y,onset_save_path,apex_save_path,flow_save_path,afflow_pic_path)
    return af


if __name__ == '__main__':
    start = time.time()
    print('cost: ', time.time() - start)
