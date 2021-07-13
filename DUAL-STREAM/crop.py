import os
import pandas as pd
import numpy as np
import dlib
import time
from PIL import Image
import cv2
from dlib68 import getxy
import scipy.io
import transplant
from limited import limits
matlab = transplant.Matlab(jvm=False, desktop=False)
def crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path,afflow_pic_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    onset_image = cv2.imread(onset_path, 1)
    apex_image = cv2.imread(apex_path, 1)

    onset_det = detector(onset_image, 0)
    apex_det = detector(apex_image, 0)
    #68点
    onex = []
    oney = []
    apex = []
    apey = []
    x1 = np.arange(68*len(onset_det)).reshape((len(onset_det), 68))
    y1 = np.arange(68*len(onset_det)).reshape((len(onset_det), 68))
    x2 = np.arange(68*len(apex_det)).reshape((len(apex_det), 68))
    y2 = np.arange(68*len(apex_det)).reshape((len(apex_det), 68))
    for i in range(len(onset_det)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(onset_image,onset_det[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])  
            x1[i][idx] = pos[0] 
            y1[i][idx] = pos[1]
    amax = 0
    for i in range(len(onset_det)):
        
        ind = 10
        dis = max(x1[i])-min(x1[i])
        if dis > amax:
            amax = dis
            inx = i      
    onex = x1[inx]
    oney = y1[inx]
    for i in range(len(apex_det)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(apex_image,apex_det[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])  
            x2[i][idx] = pos[0] 
            y2[i][idx] = pos[1]
    amax = 0
    for i in range(len(apex_det)):
        
        ind = 10
        dis = max(x2[i])-min(x2[i])
        if dis > amax:
            amax = dis
            ina = i
    apex = x2[ina]
    apey = y2[ina]
    onset_faces = dlib.full_object_detections()
    apex_faces = dlib.full_object_detections()

    onset_faces.append(predictor(onset_image, onset_det[inx]))
    apex_faces.append(predictor(apex_image, apex_det[ina]))
    sizea = 340

    onset_crops = dlib.get_face_chips(onset_image, onset_faces, size=sizea)
    apex_crops = dlib.get_face_chips(apex_image, apex_faces, size=sizea)
    onset_crop = onset_crops[0][:,:]
    apex_crop = apex_crops[0][:,:]
    
    onset_cropped = Image.fromarray(cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB))
    apex_cropped = Image.fromarray(cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB))
    
   
    onx = []
    ony = []
    apx = []
    apy = []
    #one68
    recto = detector(onset_crop, 0)
    
    landmarks = np.matrix([[p.x, p.y] for p in predictor(onset_crop,recto[0]).parts()])
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])        
        onx.append(pos[0])    
        ony.append(pos[1])
        print(idx,pos)
   
    #apex68    
    recta = detector(apex_crop, 0)
    landmarks = np.matrix([[p.x, p.y] for p in predictor(apex_crop,recta[0]).parts()])
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])        
        apx.append(pos[0])    
        apy.append(pos[1])
#    onset_crop = onset_crops[0][:280, 50:270]
#    apex_crop = apex_crops[0][:280, 50:270]
    size = 320
    
    
    xl,xr,yl,yr = limits(onx,ony,size,sizea)
    onset_crop = onset_crop[yl:yr,xl:xr]
    xl,xr,yl,yr = limits(apx,apy,size,sizea)
    apex_crop = apex_crop[yl:yr,xl:xr]
    onset_crop = cv2.resize(onset_crop,(size,size))
    apex_crop = cv2.resize(apex_crop,(size,size))
    
    onset_cropped = Image.fromarray(cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB))
    apex_cropped = Image.fromarray(cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB))
    onsets = cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB)
    apexs = cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB)
    x,y =  getxy(onsets,size)
    onset_cropped.save(onset_save_path)
    apex_cropped.save(apex_save_path)
    path_xy = onset_save_path+'xy.mat'
    scipy.io.savemat(path_xy,{'x':x, 'y':y})
    a = matlab.test(path_xy,onset_save_path,apex_save_path,flow_save_path,afflow_pic_path)
if __name__ == '__main__':
    crop_pic('s11.png','s12.png','o.jpg','a.jpg','flow.jpg','afflow.jpg')
    
    

