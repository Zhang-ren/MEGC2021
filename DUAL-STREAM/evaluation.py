import pandas as pd
ji = []
def dict_merge(test_dict1, test_dict2):
    all_test_dict = {key: test_dict1[key] + test_dict2[key] for key in test_dict1 if key in test_dict2}
    return all_test_dict

def Save_to_Csv(data, file_name, Save_format='csv', Save_type='col'):
    # data
    # 输入为一个字典，格式： { '列名称': 数据,....}
    # 列名即为CSV中数据对应的列名， 数据为一个列表

    # file_name 存储文件的名字
    # Save_format 为存储类型， 默认csv格式， 可改为 excel
    # Save_type 存储类型 默认按列存储， 否则按行存储

    # 默认存储在当前路径下

    import pandas as pd
    import numpy as np

    Name = []
    times = 0

    if Save_type == 'col':
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List).reshape(-1, 1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1, 1)))

            times += 1

        Pd_data = pd.DataFrame(columns=Name, data=Data)

    else:
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List)
            else:
                Data = np.vstack((Data, np.array(List)))

            times += 1

        Pd_data = pd.DataFrame(index=Name, data=Data)

    if Save_format == 'csv':
        Pd_data.to_csv('./' + file_name + '.csv', encoding='utf-8')
    else:
        Pd_data.to_excel('./' + file_name + '.xls', encoding='utf-8')

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
#def evalua(dataset):
#    
#    for j in range(100):
#        if dataset == 'SAMM':
#            csv_path = '6 006_1/'+str(j)+'.csv'
#        else:
#            csv_path = 'csv/'+str(j)+'.csv'
#        
#        data = pd.read_csv(csv_path, header=0)
#        pred_label = data['pred']
#        pred_tar = data['targets']
#        N,N1,N2 = 0,0,0
#        A,A1,A2,A2S,FIN_DIS = 0,0,0,[],[]
#        M,M1,M2 = 0,0,0
#        for i in range(len(pred_label)):
#            if int(pred_label[i])==0 or int(pred_label[i])==1:
#                N += 1
#                if int(pred_label[i])==0:
#                    N2 += 1
#                else:
#                    N1 += 1
#                if int(pred_tar[i])==0 or int(pred_tar[i])==1:
#                    A += 1
#                    if int(pred_tar[i]) == 0 and int(pred_label[i])==0:
#                        A2 += 1
#                    elif int(pred_tar[i]) == 1 and int(pred_label[i])==1:
#                        A1 += 1
#            if int(pred_tar[i])==0 or int(pred_tar[i])==1:
#                M += 1
#                if int(pred_tar[i]) == 0:
#                    M2 += 1
#                else:
#                    M1 += 1
#        print(N,N1,N2,A,A1,A2,M,M1,M2)
#        if A2 != 0:
#            Recall_Mae = A1 / M1
#            Precision_Mae = A1 / N1
#            F1_Mae = 2 * Recall_Mae * Precision_Mae / (Recall_Mae + Precision_Mae)
#            Recall_Me = A2 / M2
#            Precision_Me = A2 / N2
#            F1_Me = 2 * Recall_Me * Precision_Me / (Recall_Me + Precision_Me)
#            Recall_D = (A1 + A2) / (M1 + M2)
#            Precision_D = (A1 + A2) / (N1 + N2)
#            F1_D = 2 * Recall_D * Precision_D / (Recall_D + Precision_D)
#            final_dict = {'Recall_Mae': Recall_Mae, 'Precision_Mae': Precision_Mae, 'F1_Mae': F1_Mae, 'Recall_Me': Recall_Me,
#                          'Precision_Me': Precision_Me, 'F1_Me': F1_Me, 'Recall_D ': Recall_D, 'Precision_D': Precision_D,
#                          'F1_D': F1_D}
#            print(final_dict)
#            A2S.append(F1_Me)
#            ji.append(j)
#            FIN_DIS.append(final_dict)
#    F_MAX = max(A2S)
#    DIS_MAX = FIN_DIS[A2S.index(max(A2S))]
#    ij = ji[A2S.index(max(A2S))]
#    print(F_MAX,DIS_MAX,ij)
#    return ij
#e = evalua('SAMM')
#########################################
final_result = []
for i in range(100):
    csv_path = '6 006_1/'+str(i)+'.csv'
    data = pd.read_csv(csv_path, header=0)
    pred_label = data['pred']
    pred_tar = data['targets']
    subs = data['subs']
    pred_name = data['name']
    filenames = data['filel']
    N_on,N_off,N_lab,sn = {},{},{},[]
    for k in range(len(pred_label)):
        path = str(pred_name[k])
        sn.append(path)
    S_N = list(set(sn))
    S_N.sort(key=sn.index)
    n_on, n_off = [], []
    M1,M2 = 0,0
    lasti = 8748
    csv_path = '../datasets/SAMM_MEGC_optical.csv'
    datas = pd.read_csv(csv_path, header=0)
    subject = datas['subject']
    clip = datas['clip']
    clip = [i[0:5]  for i in clip]
    # clipa = [clips(i) for i in clip]
    label = datas['label']
    onset = datas['onset_frame']
    offset = datas['offset_frame']
    alls = []
    apex = datas['Apex']
    tail = datas['tail_frame']
    gt_on,gt_off,lab,GT_on,GT_off,labels = [],[],[],{},{},{}
    for i in range(len(offset)):
        if offset[i] == 0:
            offset[i] = 2 * apex[i] - onset[i] + 10
    for ss in S_N:
        gt_on, gt_off = [], []
        for s in range(len(tail)):
            path = str(clip[s])
            if path == ss:
                gt_on.append(onset[s])
                gt_off.append(offset[s])
                if int(label[s]) == 0:
                    M2 += 1
                elif int(label[s]) == 1:
                    M1 += 1
                lab.append(label[s])
            else:
                GT_on[ss] = gt_on.copy()
                GT_off[ss] = gt_off.copy()
                labels[ss] = lab
            
        GT_on[ss] = gt_on.copy()
        GT_off[ss] = gt_off.copy()
        labels[ss] = lab



    ###########################################################################

    for ss in S_N:
        n_on, n_off,n_lab = [], [], []
        for k in range(len(pred_label)):
            path = str(pred_name[k])
            if path == ss:




                if int(pred_label[k])==0 or int(pred_label[k])==1:

                    onset_name = str(filenames[k]).split('-', 2)[0]
                    offset_name = str(filenames[k]).split('-', 2)[1].split('Merge', 2)[0]
                    path = str(subs[k]+'_'+ pred_name[k])
                    n_on.append(onset_name)
                    n_off.append(offset_name)
                    n_lab.append(pred_label[k])
            else:

                N_on[ss] = n_on.copy()
                N_off[ss] = n_off.copy()
                N_lab[ss] = n_lab.copy()
                if n_on != []:
                    break
        N_on[ss] = n_on.copy()
        N_off[ss] = n_off.copy()
        N_lab[ss] = n_lab.copy()
    new_n_on,new_n_off,new_n_lab = [],[],[]
    New_N_on,New_N_off,New_N_lab = {},{},{}
    for key in N_on:
        N_on[key].append(0)
        n_f = N_on[key][0]
        new_n_on,new_n_off = [],[]
        for i in range(len(N_off[key])):
            if N_off[key][i] != N_on[key][i + 1]:
                n_l = N_off[key][i]
                new_n_on.append(n_f)
                new_n_off.append(n_l)
                new_n_lab.append(N_lab[key][i])
                n_f = N_on[key][i + 1]

        New_N_on[key] = new_n_on
        New_N_off[key] = new_n_off
        New_N_lab[key] = new_n_lab
    names,g_t_o,g_t_f,p_t_o,p_t_f,lb,p_t_l,t_l = [],[],[],[],[],[],[],[]



    for key in New_N_on:
        fn_flag = [0]*len(GT_on[key])
        for sis in range(len(New_N_on[key])):
            flag = False
            for gt_val in range(len(GT_on[key])):
                pred_range = [i for i in range(int(New_N_on[key][sis]), int(New_N_off[key][sis] )+ 1)]
                gt_range = [i for i in range(int(GT_on[key][gt_val]),int(GT_off[key][gt_val]) + 1)]
                pred_n_sla = set(pred_range).intersection(set(gt_range))
                pred_u_sla = set(pred_range).union(set(gt_range))

                for i in pred_range:
                    if i in gt_range :
                        fn_flag[gt_val] = 1
                        if len(pred_n_sla) / len(pred_u_sla) >= 0.5:
                            names.append(key)
                            g_t_o.append(GT_on[key][gt_val])
                            g_t_f.append(int(GT_off[key][gt_val]) )
                            p_t_o.append(New_N_on[key][sis])
                            p_t_f.append(int(New_N_off[key][sis]) )
                            if New_N_lab[key][sis] == 0: 
                                p_t_l.append('A2')
                                
                            elif New_N_lab[key][sis] == 1: 
                                p_t_l.append('A1')
                                
                            else:
                                p_t_l.append(' ')
                            t_l.append(' ')
                            lb.append('TP')
                        else:
                            names.append(key)
                            g_t_o.append(GT_on[key][gt_val])
                            g_t_f.append(int(GT_off[key][gt_val]) )
                            p_t_o.append(New_N_on[key][sis])
                            p_t_f.append(int(New_N_off[key][sis]) )
                            if New_N_lab[key][sis] == 0:
                                p_t_l.append('N2')                        
                            elif New_N_lab[key][sis] == 1:
                                p_t_l.append('N1')
                            if labels[key][sis] == 0: 
                                t_l.append(0)
                            else:
                                t_l.append(1)
                            
                            lb.append('FP')
                        flag = True
                        break


            if not flag:#fp

                names.append(key)
                g_t_o.append(' - ')
                g_t_f.append(' - ')
                p_t_o.append(New_N_on[key][sis])
                p_t_f.append(int(New_N_off[key][sis]))
                lb.append('FP')
                if New_N_lab[key][sis] == 0:
                    p_t_l.append('N2')
                elif New_N_lab[key][sis] == 1:
                    p_t_l.append('N1')
                if labels[key][sis] == 0: 
                    t_l.append(0)
                else:
                    t_l.append(1)

        for i in range(len(fn_flag)):
            if fn_flag[i] != 1:
                names.append(key)
                g_t_o.append(GT_on[key][i])
                g_t_f.append(int(GT_off[key][i]))
                p_t_o.append(' - ')
                p_t_f.append(' - ')
                lb.append('FN')
                if New_N_lab[key][sis] == 0:
                    p_t_l.append(' ')
                elif New_N_lab[key][sis] == 1:
                    p_t_l.append(' ')
                t_l.append(' ')
    dictsa = {'name':names,'gt_on':g_t_o,'gt_off':g_t_f,'pred_on':p_t_o,'pred_off':p_t_f,'label':lb,'Mlabel':p_t_l,'t_l':t_l}
    def PROCESS(dicts):
        data = dicts
        name = data['name']
        gt_on = data['gt_on']
        gt_off = data['gt_off']
        pred_on = data['pred_on']
        pred_off = data['pred_off']
        pred_label = data['label']
        m_label = data['Mlabel']
        for i in range(len(name)):
            for j in range(i+1,len(name)):
                if gt_on[i] != ' - ' and name[i] == name[j] and gt_on[i] == gt_on[j] and gt_off[i] == gt_off[j]  :
                    if pred_on[i] != ' ' or pred_off[i] != ' ':
                        pred_off[i] = pred_off [j]
                        pred_on[j]=' '
                        pred_off[j] = ' '
                        name[j] = ' '
                        gt_on[j] = ' '
                        gt_off[j] = ' '

                        pred_range = [i for i in range(int(pred_on[i]), int(pred_off[i]) + 1)]
                        gt_range = [i for i in range(int(gt_on[i]), int(gt_off[i]) + 1)]
                        pred_n_sla = set(pred_range).intersection(set(gt_range))
                        pred_u_sla = set(pred_range).union(set(gt_range))
                        if len(pred_n_sla) / len(pred_u_sla) >= 0.5:
                            pred_label[i] = 'TP'
                            if t_l[i] == 1 and m_label[i] == 'N1':
                                m_label[i] = 'A1'
                            elif t_l[i] == 0 and m_label[i] == 'N2':
                                m_label[i] = 'A2'
                        else:
                            pred_label[i] = 'FP'
        dict = {'name': name, 'gt_on': gt_on, 'gt_off': gt_off, 'pred_on': pred_on, 'pred_off': pred_off,'label': pred_label,'Mlabel':m_label}
        
    DICTS = PRECESS(dictsa)
    data = DICTS
    names = data['name']
    n = []
    for name in range(len(names)):
        if names[name] == ' ':
            n.append(name)

    data.drop(n,inplace = True)
    data.head()
    data.to_csv('datasamm.csv')
    csv_path = 'datasamm.csv'
    data = pd.read_csv(csv_path, header=0)
    label = data['label']
    mlabel = data['Mlabel']
    TP,FP,FN =0,0,0
    for i in label:
        if i == 'TP':
            TP += 1
        elif i == 'FP':
            FP += 1
        else:
            FN += 1
    f = 2*TP/(2*TP+FP+FN)
    print('F-score:',f)
    N1,N2,A1,A2 = 0,0,0,0
    for i in mlabel:
        if i == 'A1':
            A1 += 1
        elif i == 'A2':
            A2 += 1
        elif i == 'N1':
            N1 += 1
        elif i == 'N2':
            N2 += 1   

    
    if A1 != 0:
        Recall_Mae = A2 / M2

        Precision_Mae = A2 / N2
        F1_Mae = 2 * Recall_Mae * Precision_Mae / (Recall_Mae + Precision_Mae)
        Recall_Me = A1 / M1
        Precision_Me = A1 / N1
        F1_Me = 2 * Recall_Me * Precision_Me / (Recall_Me + Precision_Me)
        Recall_D = (A1 + A2) / (M1 + M2)
        Precision_D = (A1 + A2) / (N1 + N2)
        F1_D = 2 * Recall_D * Precision_D / (Recall_D + Precision_D)
        final_dict = {'Recall_Mae': Recall_Mae, 'Precision_Mae': Precision_Mae, 'F1_Mae': F1_Mae, 'Recall_Me': Recall_Me,
                      'Precision_Me': Precision_Me, 'F1_Me': F1_Me, 'Recall_D ': Recall_D, 'Precision_D': Precision_D,
                      'F1_D': F1_D}
        print(final_dict)
        if final_result:
            final_result.append(final_dict)
        else:
            final_result.append(dict_merge([final_result[0],final_dict]))
        Save_to_Csv(final_result,'samm_final')
            
        
