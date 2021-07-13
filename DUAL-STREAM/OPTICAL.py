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
from crop_pic import crop_pic
import seaborn as sns
import matplotlib.pyplot as plt
import random
def clips(clip):
    if 'disgust1' in clip:
        midname = '0101disgustingteeth'
    elif 'disgust2' in clip:
        midname = '0102eatingworms'
    
    elif 'anger1' in clip:
        midname = '0401girlcrashing'
    elif 'anger2' in clip:
        midname = '0402beatingpregnantwoman'
    elif 'happy1' in clip:
        midname = '0502funnyerrors'
    elif 'happy2' in clip:
        midname = '0503unnyfarting'
    elif 'happy3' in clip:
        midname = '0505funnyinnovations'
    elif 'happy4' in clip:
        midname = '0507climbingthewall'
    elif 'happy5' in clip:
        midname = '0508funnydunkey'
    return midname
def ME2_OPTICAL():
    mae = []
    me = []
    allaf = []
    csv_path = '../datasets/ME2_optical.csv'
    data = pd.read_csv(csv_path, header=0)
    subject = data['subject']
    clip = data['clip']
    clipa = [clips(i) for i in clip]
    onset = data['onset_frame']
    apex = data['Apex']
    label = data['label']
    af_path = 'allaf.csv'
    if os.path.exists(af_path):
        afdata = pd.read_csv(af_path, header=0)
        allaf = afdata['allaf']
        lens = len(allaf)-1
    else:
        lens = 0
    for index in range(lens,len(subject)):
        onset_path = '/home/halo/文档/CAM(ME)^2/rawpic/rawpic/' + subject[index] + '/' + subject[index][1:3]+ '_' + clipa[index] + '/img' + str(onset[index]).zfill(3) + '.jpg'
        print(onset_path)
        apex_path = '/home/halo/文档/CAM(ME)^2/rawpic/rawpic/' + subject[index] + '/' + subject[index][1:3]+ '_' + clipa[index] + '/img' + str(apex[index]).zfill(3) + '.jpg'
        print(apex_path)
        flow_test_path = '../ME2_flow_path/' + subject[index] + '/' + clipa[index] + '/'+clip[index]
        if not os.path.exists(flow_test_path):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(flow_test_path)
        crop_path = '../ME2_crop_path/' + subject[index] + '/' + clipa[index] + '/'  
        if not os.path.exists(crop_path):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(crop_path) 
        onset_save_path = crop_path +  '/img' + str(onset[index]).zfill(3) + '.jpg'
        apex_save_path =  crop_path + '/img' + str(apex[index]).zfill(3) + '.jpg'
        flow_save_path = flow_test_path + '/' + 'motion_flow.jpg'
        afflow_pic_path = flow_test_path + '/' + 'Merge.jpg'
        af = crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path,afflow_pic_path)
        print(af)
        if label[index] == 0:
            me.append(af)
        elif label[index] == 1:
            mae.append(af)
        allaf.append(af)
        allafs = pd.DataFrame(columns=['allaf'],data=allaf)
        allafs.to_csv('allaf.csv')
    print(np.mean(me))
    print(np.mean(mae))
def SAMM_OPTICAL():
    mae = []
    me = []
    allaf = []
    csv_path = '../datasets/SAMM_MEGC_optical.csv'
    data = pd.read_csv(csv_path, header=0)
    data.loc[328, 'offset_frame'] = data.loc[328, 'tail_frame']
    data.loc[486, 'offset_frame'] = data.loc[486, 'tail_frame']
    data = data.drop([160]).reset_index(drop=True)
    subject = data['subject']
    clip = data['clip']
    onset = data['onset_frame']
    apex = data['Apex']
    offset = data['offset_frame']
    for i in range(len(apex)):
        if apex[i] == -1:
            apex[i] = (onset[i] + offset[i]) / 2
    for i in range(len(onset)):
        if onset[i] == 0:
            onset[i] = 2*apex[i] - offset[i]
    label = data['label']
    af_path = 'samm_allaf.csv'
    if os.path.exists(af_path):
        afdata = pd.read_csv(af_path, header=0)
        allaf = list(afdata['allaf'])
        lens = len(allaf)-1
    else:
        lens = 0
    for index in range(lens,len(subject)):
        if clip[index][0:5] == '016_7':
            s = 5
        else:
            s = 4
        
        onset_path = '/home/halo/文档/SAMM_longvideos/' + clip[index][0:5] + '/' + clip[index][0:5]+ '_'+ str(onset[index]).zfill(s) + '.jpg'
        print(onset_path)
        apex_path = '/home/halo/文档/SAMM_longvideos/' + clip[index][0:5] + '/' + clip[index][0:5]+ '_' + str(apex[index]).zfill(s) + '.jpg'
        print(apex_path)
        flow_test_path = '../SAMM_flow_path/' + str(subject[index]) + '/' + clip[index] + '/'
        if not os.path.exists(flow_test_path):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(flow_test_path)
        crop_path = '../SAMM_crop_path/' + str(subject[index]) + '/' + clip[index] + '/'  
        if not os.path.exists(crop_path):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(crop_path) 
        onset_save_path = crop_path +  clip[index]+ '_'+ str(onset[index]).zfill(4) + '.jpg'
        apex_save_path =  crop_path + clip[index]+ '_' + str(apex[index]).zfill(4) + '.jpg'
        flow_save_path = flow_test_path + '/' +str(onset[index]).zfill(4) + '_'+ 'motion_flow.jpg'
        afflow_pic_path = flow_test_path + '/'+str(onset[index]).zfill(4) + '_' + 'Merge.jpg'
        af = crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path,afflow_pic_path)
        print(af)
        if label[index] == 0:
            me.append(af)
        elif label[index] == 1:
            mae.append(af)
        allaf.append(af)
        allafs = pd.DataFrame(columns=['allaf'],data=allaf)
        allafs.to_csv('samm_allaf.csv')
    print(np.mean(me))
    print(np.mean(mae))
def me2noe(Duration,tail):
    csv_path = '../datasets/ME2_optical.csv'
    data = pd.read_csv(csv_path, header=0)
    onset = data['onset_frame']
    apex = data['Apex']
    offset = data['offset_frame']
    Durations = []
    for i in range(len(onset)):
        if offset[i] != 0 and onset[i] != 0:
            Durations.append(offset[i]-onset[i])
    
    Duration = np.max(Durations)
    print(Duration)
    mae = []
    me = []
    allaf = []
    subject = data['subject']
    clip = data['clip']
    clipa = [clips(i) for i in clip]
    random.randint(0,9)
    af_path = 'no_allaf.csv'
    if os.path.exists(af_path):
        afdata = pd.read_csv(af_path, header=0)
        allaf = afdata['allaf']
        lens = len(allaf)-1
    else:
        lens = 0
    for index in range(lens,len(subject)):
        sample_path = '../datasets/BCNN3/Sample/ME2_frame/' + 'ME2_' +subject[index]+'_'+clip[index]+'.csv'
        sam_data = pd.read_csv(sample_path, header=0)
        sam_label = sam_data['label']
        sam_frame = sam_data['frame']
        for sal in range(len(sam_label)):
            if sam_label[sal] == 2 and sam_label[sal+11] == 2:
                sam_onset = sam_frame[sal]
                sam_apex = sam_frame[sal+11]
                break
              
        onset_path = '/home/halo/文档/CAM(ME)^2/rawpic/rawpic/' + subject[index] + '/' + subject[index][1:3]+ '_' + clipa[index] + '/img' + str(sam_onset).zfill(3) + '.jpg'
        print(onset_path)
        apex_path = '/home/halo/文档/CAM(ME)^2/rawpic/rawpic/' + subject[index] + '/' + subject[index][1:3]+ '_' + clipa[index] + '/img' + str(sam_apex).zfill(3) + '.jpg'
        print(apex_path)
        flow_test_path = '../ME2_flow_path/' + subject[index] + '/' + clipa[index] + '/'+clip[index]
        if not os.path.exists(flow_test_path):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(flow_test_path)
        crop_path = '../ME2_crop_path/' + subject[index] + '/' + clipa[index] + '/'  
        if not os.path.exists(crop_path):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(crop_path) 
        onset_save_path = crop_path +  '/img_sam' + str(sam_onset).zfill(3) + '.jpg'
        apex_save_path =  crop_path + '/img_sam' + str(sam_apex).zfill(3) + '.jpg'
        flow_save_path = flow_test_path + '/' + 'sam _motion_flow.jpg'
        afflow_pic_path = flow_test_path + '/' + 'sam_Merge.jpg'
        af = crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path,afflow_pic_path)
        print(af)
        
        allaf.append(af)
        allafs = pd.DataFrame(columns=['allaf'],data=allaf)
        allafs.to_csv('no_allaf.csv')
    print(np.mean(me))
    print(np.mean(mae))
def pre15_ME2flow(dua):
    csv_path = '../datasets/ME2_optical.csv'
    data = pd.read_csv(csv_path, header=0)
    s = 0
    name = []
    me = []
    allaf = []
    subject = data['subject']
    clip = data['clip']
    tail = data['tail_frame']
    clipa = [clips(i) for i in clip]
    af_path = 'per15_allaf.csv'
    lasti = 0
    if os.path.exists(af_path):
        afdata = pd.read_csv(af_path, header=0)
        allaf = afdata['allaf']
        allaf = list(allaf)
        lens = len(allaf)-1
        name = afdata['name']
        name = list(name)
        for s in range(len(tail)):
            if tail[s] == lasti:
                continue
            else:
                
                if lens - int(tail[s]/dua) > 0:
                    lens = lens -  int(tail[s]/dua)
                else:
                    lens += 1
                    s -= 1 
                    sums = lens*15+1
                    lens += 1
                    break
            lasti = tail[s]
                        
        
    else:
        lens = 1
    
    
    for i in range(s,len(tail)):
        sums = lens*dua-dua+1
        if tail[i] == lasti:
            continue
        lasti = tail[i]
        for index in range(lens*dua+1,tail[i],dua):
            print(index,clip[i])
            onset_path = '/home/halo/文档/CAM(ME)^2/rawpic/rawpic/' + subject[i] + '/' + subject[i][1:3]+ '_' + clipa[i] + '/img' + str(sums).zfill(3) + '.jpg'
            print(onset_path)
            apex_path = '/home/halo/文档/CAM(ME)^2/rawpic/rawpic/' + subject[i] + '/' + subject[i][1:3]+ '_' + clipa[i] + '/img' + str(index).zfill(3) + '.jpg'
            print(apex_path)
            flow_test_path = '../ME2_flow_all/' + subject[i] + '/' + clipa[i] + '/'
            if not os.path.exists(flow_test_path):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(flow_test_path)
            crop_path = '../ME2_crop_all/' + subject[i] + '/' + clipa[i] + '/'  
            if not os.path.exists(crop_path):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(crop_path) 
            onset_save_path = crop_path +  '/img' + str(sums).zfill(3) + '.jpg'
            apex_save_path =  crop_path + '/img' + str(index).zfill(3) + '.jpg'
            flow_save_path = flow_test_path + '/' +str(sums)+'-'+str(index).zfill(3) + 'motion_flow.jpg'
            afflow_pic_path = flow_test_path + '/' + str(sums)+'-'+str(index).zfill(3) +'Merge.jpg'
            af = crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path,afflow_pic_path)
            print(af)
            
            allaf.append(af)
            name.append(str(sums)+'-'+str(index))
            allafs = pd.DataFrame(data=allaf,index=name,columns=['allaf'] )
            allafs.to_csv('per'+dua+'_allaf.csv')
            sums = index 
            lens = 0
def pre30_SAMMflow(dua):
    allaf = []
    sa = 0
    xa = 0
    index = 0
    csv_path = '../datasets/SAMM_MEGC_optical.csv'
    data = pd.read_csv(csv_path, header=0)
    data.loc[328, 'offset_frame'] = data.loc[328, 'tail_frame']
    data.loc[486, 'offset_frame'] = data.loc[486, 'tail_frame']
    data = data.drop([160]).reset_index(drop=True)
    subject = data['subject']
    clip = data['clip']
    onset = data['onset_frame']
    apex = data['Apex']
    offset = data['offset_frame']
    for i in range(len(apex)):
        if apex[i] == -1:
            apex[i] = (onset[i] + offset[i]) / 2
    for i in range(len(onset)):
        if onset[i] == 0:
            onset[i] = 2*apex[i] - offset[i]
    label = data['label']
    
    name = []
    tail = data['tail_frame']
    af_path = 'samm_per'+dua+'_allaf.csv'
    
    lasti = 0
    if os.path.exists(af_path):
        afdata = pd.read_csv(af_path, header=0)
        allaf = afdata['allaf']
        allaf = list(allaf)
        lens = len(allaf)-1
        name = afdata['name']
        name = list(name)
        for sa in range(len(tail)):
            if tail[sa] == lasti:
                continue
            else:
                
                if lens - int(tail[sa]/dua) > 0:
                    lens = lens -  int(tail[sa]/dua)
                else:
                    lens += 1
                    sa -= 1 
                    sums = lens*dua+1
                    lens += 1
                    break
            lasti = tail[sa]
                        
        
    else:
        lens = 1

    
    
    for i in range(sa,len(tail)):
        sums = lens*dua-dua+1
        if tail[i] == lasti:
            continue
        lasti = tail[i]
        for index in range(lens*dua+1,tail[i],dua):
            if clip[i][0:5] == '016_7':
                xa = 5
            else:
                xa = 4
            print(index,clip[i])    
            onset_path = '/home/halo/文档/SAMM_longvideos/' + clip[i][0:5] + '/' + clip[i][0:5]+ '_'+ str(sums).zfill(xa) + '.jpg'
            print(onset_path)
            apex_path = '/home/halo/文档/SAMM_longvideos/' + clip[i][0:5] + '/' + clip[i][0:5]+ '_' + str(index).zfill(xa) + '.jpg'
            print(apex_path)
            flow_test_path = '../SAMM_flow_all/' + str(subject[i]) + '/' + clip[i][0:5] + '/'
            if not os.path.exists(flow_test_path):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(flow_test_path)
            crop_path = '../SAMM_crop_all/' + str(subject[i]) + '/' + clip[i][0:5] + '/'  
            if not os.path.exists(crop_path):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(crop_path) 
            onset_save_path = crop_path +  clip[i][0:5]+ '_'+ str(sums).zfill(4) + '.jpg'
            apex_save_path =  crop_path + clip[i][0:5]+ '_' + str(index).zfill(4) + '.jpg'
            flow_save_path = flow_test_path +str(sums)+'-'+str(index)+ 'motion_flow.jpg'
            afflow_pic_path = flow_test_path +str(sums)+'-'+str(index) + 'Merge.jpg'
            af = crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path,afflow_pic_path)
            print(af)
            allaf.append(af)
            allafs = pd.DataFrame(columns=['allaf'],data=allaf)
            allafs.to_csv('samm_per'+dua+'_allaf.csv')
            sums = index                          
def prepare():
    pre30_SAMMflow(30)
    pre30_SAMMflow(80)
    pre15_ME2flow(8)
    pre15_ME2flow(15)
    
    
    
    
    
    
    
    
    
        
       
if __name__ == '__main__':
    start = time.time()
    mprepare()
    

