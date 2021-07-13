import sys
sys.path.append("../Model/")
sys.path.append("../Utils/")
sys.path.append("../Train/")
import random
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from torch.autograd import Variable
from sklearn.model_selection import KFold
from torchvision import transforms
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from torch import stack, cuda, nn, optim
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#from Focal_Loss import FocalLoss
from Dataloader_Flow import Flow_loader
from Model_VGGFace import resnet18_pt_mcn, Flow_Part_npic
import argparse
import time
import torch
import dlib
import numpy as np
import warnings
import os
#from dot import make_dot
import datetime
times = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore")
file_handle=open(str(times) + '.txt',mode='w')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
random.seed(1)


super_para = {"LEARNING_RATE": 0.001, "FOLD": 95, "BATCH_SIZE": 64, "EPOCH": 100, "WEIGHT_DECAY": 0.000001,
              'Clip_Norm': 1,  'Macro': "CK",
              'Sample_File': '../Sample_File/VGG_2.txt', 'Num_Workers': 4}

LEARNING_RATE = super_para["LEARNING_RATE"]
BATCH_SIZE = super_para["BATCH_SIZE"]
EPOCH = super_para["EPOCH"]
Clip_Norm = super_para['Clip_Norm']
Sample_File = super_para['Sample_File']
FOLD = super_para['FOLD']
Num_Workers = super_para['Num_Workers']

WEIGHT_DECAY = super_para["WEIGHT_DECAY"]
if(os.path.exists(Sample_File )):
    os.remove(Sample_File)
print(super_para)
file_handle.writelines(str(super_para)+'\n')
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
file_handle.writelines(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')

parser = argparse.ArgumentParser()
parser.add_argument('--Micro_image', default='../Mix_data_22.txt')
parser.add_argument('--Micro_label', default='../SAMM_23.txt')
parser.add_argument('--Micro_subject', default='../SAMM_23.txt')
parser.add_argument('--Micro_merge', default='../SAMM_flow_23.txt')
parser.add_argument('-E', '--Epoch', default=EPOCH, type=int)


class ReverseLayerF(torch.autograd.Function):
    def __init__(self, high_value=1.0):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high_value
        self.max_iter = 10000.0

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, gradOutput):
        self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha *
                                                                           self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
        return -self.coeff * gradOutput


ReverseLayerF = ReverseLayerF()


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

def Dataset_Split(train_subjects):

    samm = []
    smic = []
    casme2 = []

    for subject in train_subjects:
        if 'samm' in subject:
            samm.append(subject)

        elif 'smic' in subject:
            smic.append(subject)

        elif 'casme2' in subject:
            casme2.append(subject)

    return samm, smic, casme2


def train(train_dataloader, model, criterion, optimizer, epoch, print_freq=7):

    model['resnet'].train()
    model['fc_top'].train()
    model['fc_bottom'].train()
    model['classifier'].train()
    model['classifier_top'].train()
    model['classifier_bottom'].train()
    model['discriminator'].train()

    correct = 0

    for i, sample in enumerate(train_dataloader):
        input, label, domain_label = sample['image'], sample['label'], sample['domain_label']
        input, label, domain_label = input.cuda(), label.cuda(), domain_label.cuda()

        _, output_resnet_top, output_resnet_bottom = model['resnet'](input)

        output_fc_top = model['fc_top'](output_resnet_top)
        output_fc_bottom = model['fc_bottom'](output_resnet_bottom)

        features = torch.cat((output_fc_top, output_fc_bottom), 1)

        reversed_features = ReverseLayerF(features)

        output_domain = model['discriminator'](reversed_features).squeeze()

        output = model['classifier'](features)
        output_top = model['classifier_top'](output_fc_top)
        output_bottom = model['classifier_bottom'](output_fc_bottom)

        loss_domain = criterion['domain'](output_domain, domain_label.float())

        loss_label = criterion['label'](output, label)
        loss_top = criterion['label'](output_top, label)
        loss_bottom = criterion['label'](output_bottom, label)

        loss = loss_label + loss_top + loss_bottom + loss_domain

        _, preds = torch.max(output, dim=1)

        correct = float((label.int() == preds.int()).sum())
        accuracy = correct / len(label)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            model['resnet'].parameters(), Clip_Norm, norm_type=2)
        nn.utils.clip_grad_norm_(
            model['fc_top'].parameters(), Clip_Norm, norm_type=2)
        nn.utils.clip_grad_norm_(
            model['fc_bottom'].parameters(), Clip_Norm, norm_type=2)
        nn.utils.clip_grad_norm_(
            model['classifier'].parameters(), Clip_Norm, norm_type=2)
        nn.utils.clip_grad_norm_(
            model['classifier_top'].parameters(), Clip_Norm, norm_type=2)
        nn.utils.clip_grad_norm_(
            model['classifier_bottom'].parameters(), Clip_Norm, norm_type=2)

        optimizer.step()

        if i % print_freq == 0:
            print('Train:\t'
                  'Epoch:[{0}][{1}/{2}]   \t'
                  'Acc: {acc:.3f}\t'
                  'Label_Loss: {l_loss:.4f}\t'
                  'Top_Loss: {t_loss:.4f}\t'
                  'Bottom_Loss: {b_loss:.4f}\t'
                  'Domain_Loss: {d_loss:.4f}\t'
                  'Loss: {loss:.4f}\t'.format(
                      epoch, i + 1, len(train_dataloader), acc=accuracy,
                      l_loss=loss_label, t_loss=loss_top, b_loss=loss_bottom, d_loss=loss_domain, loss=loss))
            file_handle.writelines('Train:\t'
                  'Epoch:[{0}][{1}/{2}]   \t'
                  'Acc: {acc:.3f}\t'
                  'Label_Loss: {l_loss:.4f}\t'
                  'Top_Loss: {t_loss:.4f}\t'
                  'Bottom_Loss: {b_loss:.4f}\t'
                  'Domain_Loss: {d_loss:.4f}\t'
                  'Loss: {loss:.4f}\t'.format(
                      epoch, i + 1, len(train_dataloader), acc=accuracy,
                      l_loss=loss_label, t_loss=loss_top, b_loss=loss_bottom, d_loss=loss_domain, loss=loss)+'\n')


def validate(validate_dataloader, model, criterion, epoch, test_subjects):

    model['resnet'].eval()
    model['fc_top'].eval()
    model['fc_bottom'].eval()
    model['classifier'].eval()
    model['classifier_top'].eval()
    model['classifier_bottom'].eval()

    losses = 0
    correctes = 0
    preds_return = torch.LongTensor([])
    target_return = torch.LongTensor([])

    sample_file = {}
    samname = []
    samtar = []
    sampred = []
    subs = []
    filel = []
    with torch.no_grad():

        for sample in validate_dataloader:
            input, target, file_name = sample['image'], sample['label'], sample['file_name']
            input, target = input.cuda(), target.cuda()
            filelname = file_name[1]
            filename = file_name[0]
            _, output_resnet_top, output_resnet_bottom = model['resnet'](input)
            output_fc_top = model['fc_top'](output_resnet_top)
            output_fc_bottom = model['fc_bottom'](output_resnet_bottom)

            output_model = torch.cat((output_fc_top, output_fc_bottom), 1)

            output = model['classifier'](output_model)

            loss = criterion['label'](output, target)

            _, preds = torch.max(output, dim=1)
            
            preds_return = torch.cat((preds_return, preds.cpu()), 0)
            
            target_return = torch.cat((target_return, target.cpu()), 0)

            losses += loss

            for f in range(len(file_name[0])):
                sample_file[file_name[0][f]] = [
                    int(preds[f].item()), int(target[f].item())]
                samname.append(file_name[0][f])
                samtar.append(target[f].item())
                sampred.append(preds[f].item())
                subs.append(str(test_subjects[0][0:3]))
                filel.append(str(filelname[f]))
                 
            
            

    return preds_return, target_return, losses, sample_file, samname, samtar, sampred, subs, filel


def build_model_2pic(num_classes=3):

    model = Flow_Part_npic(num_pic=2, num_classes=num_classes)

    model_resnet = resnet18_pt_mcn(weights_path='../Model/resnet18_pt_mcn.pth')
    pretrained_dict = model_resnet.state_dict()
    model_dict = model['resnet'].state_dict()
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

    model['resnet'].load_state_dict(model_dict)

    return model


def main():
    args = parser.parse_args()
    Micro_data = args.Micro_image
    Micro_label = args.Micro_label
    Micro_subject = args.Micro_subject
    #Micro_au = args.Micro_au
    Micro_merge = args.Micro_merge

    start_time = time.time()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomCrop((224, 224), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    with open(Micro_subject, 'r') as s:
        subject_origin = s.readlines()

    subject_origin = [subject.strip('\n') for subject in subject_origin]
    index_length = len(subject_origin)
    subject = set(subject_origin)
    subject = list(subject)
    subject.remove('Macro')
    data_array = np.arange(index_length)

    Fold_accuracy = 0

    TN_Fold = {}
    TN_Fold['samm'] = np.zeros(3, dtype=int)
    TN_Fold['smic'] = np.zeros(3, dtype=int)
    TN_Fold['casme2'] = np.zeros(3, dtype=int)
    TN_Fold['total'] = np.zeros(3, dtype=int)

    TP_Fold = {}
    TP_Fold['samm'] = np.zeros(3, dtype=int)
    TP_Fold['smic'] = np.zeros(3, dtype=int)
    TP_Fold['casme2'] = np.zeros(3, dtype=int)
    TP_Fold['total'] = np.zeros(3, dtype=int)

    FP_Fold = {}
    FP_Fold['samm'] = np.zeros(3, dtype=int)
    FP_Fold['smic'] = np.zeros(3, dtype=int)
    FP_Fold['casme2'] = np.zeros(3, dtype=int)
    FP_Fold['total'] = np.zeros(3, dtype=int)

    FN_Fold = {}
    FN_Fold['samm'] = np.zeros(3, dtype=int)
    FN_Fold['smic'] = np.zeros(3, dtype=int)
    FN_Fold['casme2'] = np.zeros(3, dtype=int)
    FN_Fold['total'] = np.zeros(3, dtype=int)

    kfold = KFold(FOLD, shuffle=True, random_state=10)
    targets = []
    predss = []
    name = []
    subss = []
    filels = []
    
    for i, (train_index, test_index) in enumerate(kfold.split(subject)):
        print('Fold: ', i)
        file_handle.writelines('Fold: ' +str(i)+'\n')
        train_subjects = [subject[i] for i in train_index]
        train_subjects.append('Macro')
        test_subjects = [subject[i] for i in test_index]
        
        print(test_subjects)
        file_handle.writelines(str(test_subjects)+'\n')


        train_index = [data_array[index] for index in range(
            index_length) if subject_origin[index] in train_subjects]
        test_index = [data_array[index] for index in range(
            index_length) if subject_origin[index] in test_subjects]
        train_dataset = Flow_loader(
            Micro_merge, Micro_label, Micro_subject, train_index, transform=transform)
        test_dataset = Flow_loader(
            Micro_merge, Micro_label, Micro_subject, test_index, transform=transform)
        

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True, num_workers=Num_Workers)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, pin_memory=True, num_workers=Num_Workers)

        model = build_model_2pic()
        

        model['discriminator'] = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(32, 1)
        )

        model['resnet'] = model['resnet'].cuda()
        model['fc_top'] = model['fc_top'].cuda()
        model['fc_bottom'] = model['fc_bottom'].cuda()
        model['fc_top'] = model['fc_top'].apply(weight_init)
        model['fc_bottom'] = model['fc_bottom'].apply(weight_init)

        model['classifier'] = model['classifier'].cuda()
        model['classifier_top'] = model['classifier_top'].cuda()
        model['classifier_bottom'] = model['classifier_bottom'].cuda()
        model['discriminator'] = model['discriminator'].cuda()

        model['classifier'] = model['classifier'].apply(weight_init)
        model['classifier_top'] = model['classifier_top'].apply(weight_init)
        model['classifier_bottom'] = model['classifier_bottom'].apply(
            weight_init)
        model['discriminator'] = model['discriminator'].apply(weight_init)

        criterion = {}
        criterion['label'] = torch.nn.CrossEntropyLoss()
        criterion['domain'] = torch.nn.BCEWithLogitsLoss()

        optimizer = optim.Adam([
            {'params': model['resnet'].parameters(), 'lr':0.00001},
            {'params': model['fc_top'].parameters(), 'lr':LEARNING_RATE},
            {'params': model['fc_bottom'].parameters(), 'lr':LEARNING_RATE},
            {'params': model['classifier'].parameters(), 'lr':LEARNING_RATE},
            {'params': model['classifier_top'].parameters(),
             'lr':LEARNING_RATE},
            {'params': model['classifier_bottom'].parameters(),
             'lr':LEARNING_RATE},
            {'params':model['discriminator'].parameters(), 'lr':LEARNING_RATE},
        ], weight_decay=WEIGHT_DECAY)

        Epoch_accuracy = 0
        Epoch_F1_score = 0
        Epoch_Recall = 0

        
        for epoch in range(args.Epoch):
            targets = []
            predss = []
            name = []
            subss = []
            filels = []
            if epoch % 10 == 0 and epoch != 0:
                if optimizer.param_groups[2]['lr'] > 0.00001:
                    optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] * 0.5
                    optimizer.param_groups[2]['lr'] = optimizer.param_groups[2]['lr'] * 0.5
                    optimizer.param_groups[3]['lr'] = optimizer.param_groups[3]['lr'] * 0.5
                    optimizer.param_groups[4]['lr'] = optimizer.param_groups[4]['lr'] * 0.5
                    optimizer.param_groups[5]['lr'] = optimizer.param_groups[5]['lr'] * 0.5
                    optimizer.param_groups[6]['lr'] = optimizer.param_groups[6]['lr'] * 0.5

            train(train_loader, model, criterion, optimizer, epoch)
            preds, target, loss, temp_file, samname, samtar, sampred, subs, filel = validate(test_loader, model, criterion, epoch, test_subjects)
            name.extend(samname)
            predss.extend(sampred)
            targets.extend(samtar)
            subss.extend(subs)
            filels.extend(filel)
            
            
            sadaa = {'name':name,'pred':predss, 'targets':targets,'subs':subss,'filel':filels}
            allafs = pd.DataFrame(data=sadaa )
            if not os.path.exists('result_csv/' +str(test_subjects[0]) +'/'):
                os.makedirs('result_csv/' +str(test_subjects[0]) +'/')
                
            allafs.to_csv('result_csv/' +str(test_subjects[0]) +'/'+ str(epoch) +'.csv')
            accuracy = accuracy_score(target, preds)
            print(accuracy)

            losses = loss / len(test_index)
            
    super_para = {"LEARNING_RATE": 0.001, "FOLD": 141, "BATCH_SIZE": 64, "EPOCH": 100, "WEIGHT_DECAY": 0.000001,
              'Clip_Norm': 1,  'Macro': "SAMM,CASME,SMIC",
              'Sample_File': '../Sample_File/VGG_2.txt', 'Num_Workers': 4}

    LEARNING_RATE = super_para["LEARNING_RATE"]
    BATCH_SIZE = super_para["BATCH_SIZE"]
    EPOCH = super_para["EPOCH"]
    Clip_Norm = super_para['Clip_Norm']
    Sample_File = super_para['Sample_File']
    FOLD = super_para['FOLD']
    Num_Workers = super_para['Num_Workers']

    WEIGHT_DECAY = super_para["WEIGHT_DECAY"]
    if(os.path.exists(Sample_File )):
        os.remove(Sample_File)
    print(super_para)
    file_handle.writelines(str(super_para)+'\n')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    file_handle.writelines(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('--Micro_image', default='../ME2_data_22.txt')
    parser.add_argument('--Micro_label', default='../ME2_23.txt')
    parser.add_argument('--Micro_subject', default='../ME2_23.txt')
    parser.add_argument('--Micro_merge', default='../ME2_flow_23.txt')
    parser.add_argument('-E', '--Epoch', default=EPOCH, type=int)

    
    
            
    args = parser.parse_args()
    Micro_data = args.Micro_image
    Micro_label = args.Micro_label
    Micro_subject = args.Micro_subject
    #Micro_au = args.Micro_au
    Micro_merge = args.Micro_merge

    start_time = time.time()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomCrop((224, 224), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    with open(Micro_subject, 'r') as s:
        subject_origin = s.readlines()

    subject_origin = [subject.strip('\n') for subject in subject_origin]
    index_length = len(subject_origin)
    subject = set(subject_origin)
    subject = list(subject)
    subject.remove('Macro')
    data_array = np.arange(index_length)

    Fold_accuracy = 0

    TN_Fold = {}
    TN_Fold['samm'] = np.zeros(3, dtype=int)
    TN_Fold['smic'] = np.zeros(3, dtype=int)
    TN_Fold['casme2'] = np.zeros(3, dtype=int)
    TN_Fold['total'] = np.zeros(3, dtype=int)

    TP_Fold = {}
    TP_Fold['samm'] = np.zeros(3, dtype=int)
    TP_Fold['smic'] = np.zeros(3, dtype=int)
    TP_Fold['casme2'] = np.zeros(3, dtype=int)
    TP_Fold['total'] = np.zeros(3, dtype=int)

    FP_Fold = {}
    FP_Fold['samm'] = np.zeros(3, dtype=int)
    FP_Fold['smic'] = np.zeros(3, dtype=int)
    FP_Fold['casme2'] = np.zeros(3, dtype=int)
    FP_Fold['total'] = np.zeros(3, dtype=int)

    FN_Fold = {}
    FN_Fold['samm'] = np.zeros(3, dtype=int)
    FN_Fold['smic'] = np.zeros(3, dtype=int)
    FN_Fold['casme2'] = np.zeros(3, dtype=int)
    FN_Fold['total'] = np.zeros(3, dtype=int)

    kfold = KFold(FOLD, shuffle=True, random_state=10)
    targets = []
    predss = []
    name = []
    subss = []
    filels = []
    
    for i, (train_index, test_index) in enumerate(kfold.split(subject)):
        print('Fold: ', i)
        file_handle.writelines('Fold: ' +str(i)+'\n')
        train_subjects = [subject[i] for i in train_index]
        train_subjects.append('Macro')
        test_subjects = [subject[i] for i in test_index]
        
        print(test_subjects)
        file_handle.writelines(str(test_subjects)+'\n')


        train_index = [data_array[index] for index in range(
            index_length) if subject_origin[index] in train_subjects]
        test_index = [data_array[index] for index in range(
            index_length) if subject_origin[index] in test_subjects]
        train_dataset = Flow_loader(
            Micro_merge, Micro_label, Micro_subject, train_index, transform=transform)
        test_dataset = Flow_loader(
            Micro_merge, Micro_label, Micro_subject, test_index, transform=transform)
        

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True, num_workers=Num_Workers)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, pin_memory=True, num_workers=Num_Workers)

        model = build_model_2pic()
        

        model['discriminator'] = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(32, 1)
        )

        model['resnet'] = model['resnet'].cuda()
        model['fc_top'] = model['fc_top'].cuda()
        model['fc_bottom'] = model['fc_bottom'].cuda()
        model['fc_top'] = model['fc_top'].apply(weight_init)
        model['fc_bottom'] = model['fc_bottom'].apply(weight_init)

        model['classifier'] = model['classifier'].cuda()
        model['classifier_top'] = model['classifier_top'].cuda()
        model['classifier_bottom'] = model['classifier_bottom'].cuda()
        model['discriminator'] = model['discriminator'].cuda()

        model['classifier'] = model['classifier'].apply(weight_init)
        model['classifier_top'] = model['classifier_top'].apply(weight_init)
        model['classifier_bottom'] = model['classifier_bottom'].apply(
            weight_init)
        model['discriminator'] = model['discriminator'].apply(weight_init)

        criterion = {}
        criterion['label'] = torch.nn.CrossEntropyLoss()
        criterion['domain'] = torch.nn.BCEWithLogitsLoss()

        optimizer = optim.Adam([
            {'params': model['resnet'].parameters(), 'lr':0.00001},
            {'params': model['fc_top'].parameters(), 'lr':LEARNING_RATE},
            {'params': model['fc_bottom'].parameters(), 'lr':LEARNING_RATE},
            {'params': model['classifier'].parameters(), 'lr':LEARNING_RATE},
            {'params': model['classifier_top'].parameters(),
             'lr':LEARNING_RATE},
            {'params': model['classifier_bottom'].parameters(),
             'lr':LEARNING_RATE},
            {'params':model['discriminator'].parameters(), 'lr':LEARNING_RATE},
        ], weight_decay=WEIGHT_DECAY)

        Epoch_accuracy = 0
        Epoch_F1_score = 0
        Epoch_Recall = 0

        
        for epoch in range(args.Epoch):
            targets = []
            predss = []
            name = []
            subss = []
            filels = []
            if epoch % 10 == 0 and epoch != 0:
                if optimizer.param_groups[2]['lr'] > 0.00001:
                    optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] * 0.5
                    optimizer.param_groups[2]['lr'] = optimizer.param_groups[2]['lr'] * 0.5
                    optimizer.param_groups[3]['lr'] = optimizer.param_groups[3]['lr'] * 0.5
                    optimizer.param_groups[4]['lr'] = optimizer.param_groups[4]['lr'] * 0.5
                    optimizer.param_groups[5]['lr'] = optimizer.param_groups[5]['lr'] * 0.5
                    optimizer.param_groups[6]['lr'] = optimizer.param_groups[6]['lr'] * 0.5

            train(train_loader, model, criterion, optimizer, epoch)
            preds, target, loss, temp_file, samname, samtar, sampred, subs, filel = validate(test_loader, model, criterion, epoch, test_subjects)
            name.extend(samname)
            predss.extend(sampred)
            targets.extend(samtar)
            subss.extend(subs)
            filels.extend(filel)
            
            
            sadaa = {'name':name,'pred':predss, 'targets':targets,'subs':subss,'filel':filels}
            allafs = pd.DataFrame(data=sadaa )
            if not os.path.exists('result_csv/' +str(test_subjects[0]) +'/'):
                os.makedirs('result_csv/' +str(test_subjects[0]) +'/')
                
            allafs.to_csv('result_csv/' +str(test_subjects[0]) +'/'+ str(epoch) +'.csv')
            accuracy = accuracy_score(target, preds)
            print(accuracy)

            losses = loss / len(test_index)



    
   

    end_time = time.time()
    print('Total cost time: ', end_time - start_time)
 
    file_handle.close()

if __name__ == '__main__':
    main()
