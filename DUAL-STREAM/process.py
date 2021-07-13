import numpy as np
import pandas as pd
import os
Expression_version = 'SAMM'
Version = '23'

LABEL_FILE = 'MEGC2021/{}_label_{}.txt'.format(Expression_version, Version)
SUBJECT_FILE = 'MEGC2021/{}_subject_{}.txt'.format(Expression_version, Version)
FLOW_FILE = 'MEGC2021/{}_flow_{}.txt'.format(Expression_version, Version)
if(os.path.exists(LABEL_FILE )):
   os.remove(LABEL_FILE)
if(os.path.exists(SUBJECT_FILE)):
   os.remove(SUBJECT_FILE)
if(os.path.exists(FLOW_FILE)):
   os.remove(FLOW_FILE)
NAMES = ['ME2','SAMM' ,'crop','concat']
NAME = NAMES[3]
if NAME == 'ME2':
    csv_path = 'OpticalFlow/ME2_LABEL.csv'
    Flow_pic_paths = '../../ME2_flow_all/'
if NAME == 'SAMM':
    csv_path = 'OpticalFlow/SAMM_LABEL.csv'
    Flow_pic_paths = '../../SAMM_flow_all/'
if NAME == 'crop':
    csv_path = 'OpticalFlow/SAMM_LABEL50.csv'
    Flow_pic_paths = '../../SAMM_flow_all/'
if NAME == 'concat':
    csv_path = 'OpticalFlow/SAMM_LABEL50.csv'
    Flow_pic_paths = '../../SAMM_flow_all/'

data = pd.read_csv(csv_path, header=0)
sub  = data['sub']
clip = data['clip']
label = data['label']
name = data['name']

for i in range(len(sub)):
    Flow_pic_path = Flow_pic_paths + str(sub[i]) +'/'+ str(clip[i]) +'/'+ name[i]

    with open (FLOW_FILE, 'a') as m:
        m.write(Flow_pic_path + '\n')
    with open (LABEL_FILE, 'a') as l:
        l.write(str(label[i]) + '\n')
    with open (SUBJECT_FILE, 'a') as s:
        s.write(str(sub[i]) +' '+ str(clip[i]) + '\n')
if NAME == 'concat':
    csv_path = 'OpticalFlow/ME2_LABEL.csv'
    Flow_pic_paths = '../../ME2_flow_all/'
    data = pd.read_csv(csv_path, header=0)
    sub = data['sub']
    clip = data['clip']
    label = data['label']
    name = data['name']
    for i in range(len(sub)):
        Flow_pic_path = Flow_pic_paths + str(sub[i]) + '/' + str(clip[i]) + '/' + str(name[i])

        with open (FLOW_FILE, 'a') as m:
            m.write(Flow_pic_path + '\n')
        with open (LABEL_FILE, 'a') as l:
            l.write(str(label[i]) + '\n')
        with open (SUBJECT_FILE, 'a') as s:
            s.write(str(sub[i]) +' '+ str(clip[i]) + '\n')

if NAME == 'crop':
    csv_path = 'OpticalFlow/ME2_crop_LABEL.csv'
    Flow_pic_paths = '../../ME2_flow_path/'
    data = pd.read_csv(csv_path, header=0)
    sub = data['sub']
    clip = data['clip']
    label = data['label']
    name = data['name']

    for i in range(len(sub)):
        Flow_pic_path = Flow_pic_paths + str(sub[i]) + '/' + str(clip[i]) + '/' + str(name[i])

        with open (FLOW_FILE, 'a') as m:
            m.write(Flow_pic_path + '\n')
        with open (LABEL_FILE, 'a') as l:
            l.write(str(label[i]) + '\n')
        with open (SUBJECT_FILE, 'a') as s:
            s.write(str(sub[i]) +' '+ str(clip[i]) + '\n')


c = open('CK_PATH.txt')
sc = c.read().splitlines()

for i in range(len(sc)):
    with open (FLOW_FILE, 'a') as m:
        m.write(sc[i] + '\n')
    with open (LABEL_FILE, 'a') as l:
        l.write(str(1) + '\n')
    with open (SUBJECT_FILE, 'a') as s:
        s.write(str('Macro') + '\n')
c.close()
me = open('MICRO.txt')
sm = me.read().splitlines()
for i in range(len(sc)):
    with open (FLOW_FILE, 'a') as m:
        m.write(sm[i] + '\n')
    with open (LABEL_FILE, 'a') as l:
        l.write(str(0) + '\n')
    with open (SUBJECT_FILE, 'a') as s:
        s.write(str('Macro') + '\n')
me.close()

