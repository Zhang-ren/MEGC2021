import pandas as pd
import os

import numpy as np
from PIL import Image
import cv2
from dlib68 import getxy
from torchvision import transforms
from torch import nn
from apexs import imflow

def trans_square(image):
    r"""transform square.
    :return PIL image
    """
    img = transforms.ToTensor()(image)
    C, H, W = img.shape
    pad_1 = int(abs(H - W) // 2)  # 一侧填充长度
    pad_2 = int(abs(H - W) - pad_1)  # 另一侧填充长度
    img = img.unsqueeze(0)  # 加轴
    if H > W:
        img = nn.ZeroPad2d((pad_1, pad_2, 0, 0))(img)  # 左右填充，填充值是0
        # img = nn.ConstantPad2d((pad_1, pad_2, 0, 0), 127)(img)  # 左右填充，填充值是127
    elif H < W:
        img = nn.ZeroPad2d((0, 0, pad_1, pad_2))(img)  # 上下填充，填充值是0
        # img = nn.ConstantPad2d((0, 0, pad_1, pad_2), 127)(img)  # 上下填充，填充值是127
    img = img.squeeze(0)  # 减轴
    img = transforms.ToPILImage()(img)
    return img
def crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path, afflow_pic_path):
    onset_image = cv2.imread(onset_path, 1)
    apex_image= cv2.imread(apex_path, 1)
    onset_crop = trans_square(onset_image)
    apex_crop = trans_square(apex_image)

    onset_cropped = Image.fromarray(cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB))
    apex_cropped = Image.fromarray(cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB))
    onset_cropped.save(onset_save_path)
    apex_cropped.save(apex_save_path)
    onsets = cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB)
    apexs = cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB)
    x ,y = [],[]

    af = imflow(x,y,onset_save_path, apex_save_path, flow_save_path, afflow_pic_path)
    return af
csv_path = 'ME2_optical.csv'
data = pd.read_csv(csv_path, header=0)
s = 0
name = []
me = []
allaf = []
subject = data['subject']
clip = data['clip']
tail = data['tail_frame']
af_path = 'per15_allaf.csv'
lasti = 0


for i in range(len(tail)):
    sums = 1
    if tail[i] == lasti:
        continue
    lasti = tail[i]
    for index in range(1, tail[i], 15):
        print(index, clip[i])
        onset_path = '../CAM(ME)^2/cropped/' + subject[i][1:] + '/' + clip[
            i] + '/img' + str(sums).zfill(3) + '.jpg'
        print(onset_path)
        apex_path = '../CAM(ME)^2/cropped/' + subject[i][1:] + '/' + '_' + clip[
            i] + '/img' + str(index).zfill(3) + '.jpg'
        print(apex_path)
        flow_test_path = '../ME2_flow_all/' + subject[i][1:] + '/' + clip[i] + '/'
        if not os.path.exists(flow_test_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(flow_test_path)
        crop_path = '../ME2_crop_all/' + subject[i] + '/' + clip[i] + '/'
        if not os.path.exists(crop_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(crop_path)
        onset_save_path = crop_path + '/img' + str(sums).zfill(3) + '.jpg'
        apex_save_path = crop_path + '/img' + str(index).zfill(3) + '.jpg'
        flow_save_path = flow_test_path + '/' + str(sums) + '-' + str(index).zfill(3) + 'motion_flow.jpg'
        afflow_pic_path = flow_test_path + '/' + str(sums) + '-' + str(index).zfill(3) + 'Merge.jpg'
        af = crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path, afflow_pic_path)
        print(af)

        allaf.append(af)
        name.append(str(sums) + '-' + str(index))
        allafs = pd.DataFrame(data=allaf, index=name, columns=['allaf'])
        allafs.to_csv('per15_allaf.csv')
        sums = index
        lens = 0
