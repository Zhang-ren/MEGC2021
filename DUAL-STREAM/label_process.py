import pandas as pd
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
def fixlabel():
    fix_path = 'fix_per15_allaf.csv'
    fix_data = pd.read_csv(fix_path, header=0)
    fix_af = list(fix_data['allaf'])
    fix_name = list(fix_data['name'])
    fix1_path = 'fix1_per15_allaf.csv'
    fix1_data = pd.read_csv(fix1_path, header=0)
    fix1_af = list(fix1_data['allaf'])
    fix1_name = list(fix1_data['name'])
    af_path = 'per15_allaf.csv'
    af =  pd.read_csv(af_path, header=0)
    allaf = list(af['allaf'])
    names = list(af['name'])
    ccc, save_onset, ccs = ME2_process()
    while ccc != save_onset:
        ccc, save_onset, ccs = ME2_process()
        for i in range(len(ccc)):
            print('error')
            if ccc[i] != save_onset[i]:
                print(ccs[i])
                print('ans:{},pre:{}'.format(ccc[i], save_onset[i]))
                for si in range(17):
                    names.insert(i + si, fix_name.pop(0))
                    allaf.insert(i + si, fix_af.pop(0))
                names.insert(i + 17, fix1_name.pop(0))

                allaf.insert(i + 17, fix1_af.pop(0))
                print(i)

                break
    else:
        print('done')


               

        di = {'name': names, 'allaf': allaf}
        allafs = pd.DataFrame(data=di)
        allafs.to_csv('per15_allaf.csv')
def ME2_process(dua=15):
    csv_path = '../datasets/ME2_optical.csv'
    data = pd.read_csv(csv_path, header=0)
    subject = data['subject']
    clip = data['clip']
    tail = data['tail_frame']
    clipa = [clips(i) for i in clip]
    af_path = 'per15_allaf.csv'
    onset = data['onset_frame']
    offset = data['offset_frame']
    label = data['label']
    apex = data['Apex']
    for i in range(len(offset)):
        if offset[i] == 0:
            offset[i] = 2 * apex[i] - onset[i] +10
    subj = []
    cli = []
    ons = []
    alls = []
    ori_label = []
    all_label = []
    save_dict = {}
    lasti = 3711
    for s in range(len(tail)):
        if tail[s] == lasti:
            a = [frame for frame in range(onset[s]-8,offset[s]-6)]
            ons.append(a)
            ori_label.append(label[s])
            continue
        else:

            alls.append(ons)
            all_label.append(ori_label)
            ons = []
            ori_label = []
            a = [frame for frame in range(onset[s] - 8, offset[s] - 6)]
            ons.append(a)
            ori_label.append(label[s])
        lasti = tail[s]
    alls.append(ons)
    all_label.append(ori_label)


    af =  pd.read_csv(af_path, header=0)
    allaf = list(af['allaf'])
    names = list(af['name'])
    save_name = []
    save_onset = []
    save_offset = []
    save_label = []
    for name_index in range(len(names)):
        tmp_name = str(names[name_index]).split('-', 2)
        save_name.append(tmp_name[0] + '-' + tmp_name[1].zfill(3) + 'Merge.jpg')
        save_onset.append(tmp_name[0].zfill(3))
        save_offset.append(tmp_name[1].zfill(3))

    lasti = 0
    c = 0
    qqq = 0
    ccc = []
    ccs = []
    new_all = []

    for s in range(len(tail)):
            if tail[s] == lasti:

                continue
            else:

                for i in range(int(tail[s]/dua)):
                    if (i+1) * dua + 1 < tail[s]:

                        ccc.append(str((i) * dua +1).zfill(3))
                        ccs.append(str(subject[s]+clip[s]))
                        qqq += 1
                        subj.append(subject[s])
                        cli.append(clipa[s])
                        name_num = i * dua+1

                        if c == 1:
                            print(int(name_num))
                            #print(int(tail[s]))
                        for sla, slabel in zip(alls[c], all_label[c]):
                            # print(sla)
                            #
                            # print(slabel)


                            if int(name_num) in sla:
                               # print(10000)
                                save_label.append(slabel)
                                break

                        else:
                            save_label.append(2)


                c += 1


            lasti = tail[s]

    print(len(subj),len(cli),len(save_name),len(save_onset),len(save_offset),len(save_label),len(allaf))

    save_dict = {'sub':subj,'clip':cli,'name':save_name,'onset_frame':save_onset,'offset_frame':save_offset,'label':save_label,'allaf':allaf}
    allafs = pd.DataFrame(data=save_dict)
    allafs.to_csv('ME2_LABEL.csv')
    return ccc,save_onset ,ccs
def samm_process(dua=30):
    csv_path = '../datasets/SAMM_MEGC_optical.csv'
    data = pd.read_csv(csv_path, header=0)
    data.loc[328, 'offset_frame'] = data.loc[328, 'tail_frame']
    data.loc[486, 'offset_frame'] = data.loc[486, 'tail_frame']
    data = data.drop([160]).reset_index(drop=True)
    subject = data['subject']
    clip = data['clip']
    tail = data['tail_frame']
    clipa = [i[0:5] for i in clip]
    af_path = 'samm_per15_allaf.csv'
    onset = data['onset_frame']
    offset = data['offset_frame']
    label = data['label']
    apex = data['Apex']
    for i in range(len(apex)):
        if apex[i] == -1:
            apex[i] = (onset[i] + offset[i]) / 2
    for i in range(len(onset)):
        if onset[i] == 0:
            onset[i] = 2*apex[i] - offset[i]
    for i in range(len(offset)):
        if offset[i] == 0:
            offset[i] = 2 * apex[i] - onset[i] +10
    subj = []
    cli = []
    ons = []
    alls = []
    ori_label = []
    all_label = []
    save_dict = {}
    lasti = 8748
    for s in range(len(tail)):
        if tail[s] == lasti:
            a = [frame for frame in range(onset[s]-16,offset[s]-15)]
            ons.append(a)
            ori_label.append(label[s])
            continue
        else:

            alls.append(ons)
            all_label.append(ori_label)
            ons = []
            ori_label = []
            a = [frame for frame in range(onset[s] - 16, offset[s] - 15)]
            ons.append(a)
            ori_label.append(label[s])
        lasti = tail[s]
    alls.append(ons)
    all_label.append(ori_label)


    af =  pd.read_csv(af_path, header=0)
    allaf = list(af['allaf'])
    save_name = []
    save_onset = []
    save_offset = []
    save_label = []
#    for name_index in range(len(names)):
#        save_name.append(tmp_name[0] + '-' + tmp_name[1].zfill(3) + 'Merge.jpg')
#        save_onset.append(tmp_name[0].zfill(3))
#        save_offset.append(tmp_name[1].zfill(3))

    lasti = 0
    c = 0
    qqq = 0
    ccc = []
    ccs = []
    new_all = []
    for s in range(len(tail)):
            if tail[s] == lasti:

                continue
            else:

                for i in range(int(tail[s]/dua)):
                    if (i+1) * dua + 1 < tail[s]:
                        print(len(alls))
                        save_onset.append(str((i) * dua + 1).zfill(4))
                        save_offset.append(str((i+1) * dua + 1).zfill(4))
                        save_name.append(str((i) * dua + 1) + '-' +str((i+1) * dua + 1) + 'Merge.jpg')
                        ccc.append(str((i) * dua+1).zfill(3))
                        ccs.append(str(subject[s]) +str( clipa[s]))
                        qqq += 1
                        subj.append(subject[s])
                        cli.append(clipa[s])
                        name_num = i * dua+1

                        
                        for sla, slabel in zip(alls[c], all_label[c]):
                            # print(sla)
                            #
                            # print(slabel)


                            if int(name_num) in sla:
                               # print(10000)
                                save_label.append(slabel)
                                break

                        else:
                            save_label.append(2)


                c += 1


            lasti = tail[s]

    print(len(subj),len(cli),len(save_name),len(save_onset),len(save_offset),len(save_label),len(allaf))

    save_dict = {'sub':subj,'clip':cli,'name':save_name,'onset_frame':save_onset,'offset_frame':save_offset,'label':save_label}
    allafs = pd.DataFrame(data=save_dict)
    allafs.to_csv('SAMM_LABEL'+str(dua)+'.csv')
    return ccc,save_onset ,ccs
def ME2_crop():
    csv_path = '../datasets/ME2_optical.csv'
    data = pd.read_csv(csv_path, header=0)
    subject = data['subject']
    clip = data['clip']
    save_name = []
    clipa = [clips(i) for i in clip]

    onset = data['onset_frame']
    offset = data['offset_frame']
    label = data['label']
    apex = data['Apex']
    for i in range(len(offset)):
        if offset[i] == 0:
            offset[i] = 2 * apex[i] - onset[i] +10
    for i in range(len(subject)):
        save_name.append('Merge.jpg')
    clip = [clipa[i] + '/' +clip[i] for i in range(len(clip))]

    save_dict = {'sub':subject,'clip':clip,'name':save_name,'onset_frame':onset,'offset_frame':offset,'label':label}
    allafs = pd.DataFrame(data=save_dict)
    allafs.to_csv('ME2_crop_LABEL.csv')
if __name__ == '__main__':
    
    ccc,save_onset ,ccs = samm_process(80)
    ccc,save_onset ,ccs = samm_process()
    ccc,save_onset ,ccs = ME2_process(dua=15)
    ccc,save_onset ,ccs = ME2_process(dua=8)
    ME2_crop()




