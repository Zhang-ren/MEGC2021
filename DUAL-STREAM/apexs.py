import os
import dlib
import cv2
import transplant
import numpy as np
import operator
from dlib68 import getxy
matlab = transplant.Matlab(jvm=False, desktop=False)
def crop_faces(onset_path,apex_path,sizea = 340):
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
    onset_crops = dlib.get_face_chips(onset_image, onset_faces, size=sizea)
    apex_crops = dlib.get_face_chips(apex_image, apex_faces, size=sizea)
    onset_crop = onset_crops[0]
    apex_crop = apex_crops[0]
#    onx = []
#    ony = []
#    apx = []
#    apy = []
#    #on68
#    recto = detector(onset_crop, 0)
#    
#    landmarks = np.matrix([[p.x, p.y] for p in predictor(onset_crop,recto[0]).parts()])
#    for idx, point in enumerate(landmarks):
#        pos = (point[0, 0], point[0, 1])        
#        onx.append(pos[0])    
#        ony.append(pos[1])
#        
#   
#    #apex68    
#    recta = detector(apex_crop, 0)
#    
#    landmarks = np.matrix([[p.x, p.y] for p in predictor(apex_crop,recta[0]).parts()])
#    for idx, point in enumerate(landmarks):
#        pos = (point[0, 0], point[0, 1])        
#        apx.append(pos[0])    
#        apy.append(pos[1])
    return onset_crop,apex_crop #,sizea,onx,ony,apx,apy
def crop_face(onset_path, apex_path,  sizea):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    onset_image = cv2.imread(onset_path, 1)
    apex_image = cv2.imread(apex_path, 1)

    onset_det = detector(onset_image, 1)[0]
    apex_det = detector(apex_image, 1)[0]

    onset_faces = dlib.full_object_detections()
    apex_faces = dlib.full_object_detections()

    onset_faces.append(predictor(onset_image, onset_det))
    apex_faces.append(predictor(apex_image, apex_det))

    onset_crops = dlib.get_face_chips(onset_image, onset_faces, size=sizea)
    apex_crops = dlib.get_face_chips(apex_image, apex_faces, size=sizea)
    onset_crop = onset_crops[0]
    apex_crop = apex_crops[0]
    return onset_crop,apex_crop
def apexs(test_image_path,image_dir,mode=0,find=0):
    rads = []
    onset_path = os.path.join(test_image_path, image_dir[0])
    if mode == 0:
        for i in range(1,len(image_dir)):
            apex_path = os.path.join(test_image_path, image_dir[i])
            onset_crop,apex_crop,size,_,_,_,_ = crop_face(onset_path,apex_path)
            ox,oy = getxy(onset_crop,size)
            ox = np.array(ox)
            oy = np.array(oy)
            rad = matlab.maxflow(onset_crop,apex_crop,ox,oy,0)
            rads.append(rad)
        max_index, max_number = max(enumerate(rads), key=operator.itemgetter(1))
        
        return max_index
    else:
        flag = 0
        used_apex = {}
        low = 1 #数组最小索引值
        high = int(len(image_dir)) - 1 #数组最大索引值
        while low <= high:
            apex_num = int((low + high) / 2)
            apex_path =os.path.join(test_image_path, image_dir[apex_num])
            onset_crop,apex_crop,size,_,_,_,_ = crop_face(onset_path,apex_path,320)
            ox,oy = getxy(onset_crop,size)
            ox = np.array(ox)
            oy = np.array(oy)
            rad = matlab.maxflow(onset_crop,apex_crop,ox,oy,0)
            lr = 8
            hr = 11
            if rad >= lr and rad <= hr:
                flag = 1                
                print(rad,'find')
                return apex_num
                break
            elif rad < lr:	        
                low = apex_num + 1
                used_apex[apex_num] = abs(rad - (lr+hr)/2)
            else:
                high = apex_num - 1
                used_apex[apex_num] = abs(rad - (lr+hr)/2)
        if flag == 0:
            if find == 0:
                print(rad)
                return apex_num
            else:
                min_key = apex_num
                for key in used_apex.keys():
                    if used_apex[key] < used_apex[min_key]:
                        min_key = key 
                print(used_apex[min_key]+(lr+hr)/2)
                return min_key
def imflow(x,y,onset_save_path,apex_save_path,flow_save_path,afflow_pic_path):
    if x and y:
        while min(x) < 0:
            x[list(x).index(min(x))] = 0
        while max(x) >=399 :
            x[list(x).index(max(x))] = 398
        while min(y) < 0:
            y[list(y).index(min(y))] = 0
        while max(y) >= 399:
            y[list(y).index(max(y))] = 398
    
    maxrad = matlab.test(x,y,onset_save_path,apex_save_path,flow_save_path,afflow_pic_path)
    #onset_crop,apex_crop = crop_face(onset_save_path,apex_save_path,320)
#    ox,oy = getxy(onset_crop,340)
#    print(ox,oy)
#    onset_crop = cv2.imread(onset_save_path, 1)
#    apex_crop = cv2.imread(apex_save_path, 1)
#    ox = np.array(x)
#    oy = np.array(y)
#    rad = matlab.maxflow(onset_crop,apex_crop,ox,oy,0)
#    print('maxrad = {},rad = {}'.format(maxrad,rad))
    return maxrad
    
             
            
	        

        
        
if __name__ == '__main__' :
    test_image_path = '/home/halo/音乐/MEGC2019-TIMED/CK+/cohn-kanade-images/S005/001'
    image_dir = os.listdir(test_image_path)
    image_dir = [item for item in image_dir if os.path.splitext(item)[1] == '.png']
    image_dir = sorted(image_dir)
    s =  apexs(test_image_path,image_dir,1,1)
    print(s)
    
    
    
    
    
    
    
    
