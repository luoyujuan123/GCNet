import os
import numpy as np
with open('/media/hd0/liujiayu/data/ufc101/contain/ucf101_train_split_1_rawframes.txt','r') as f:
    lines = f.readlines()

import numpy as np
# ranges=np.zeros(51, dtype=np.int32)
ranges=[str(i) for i in range(101)]
ranges[3]='3sf'
# print(ranges)
for line in lines:
    line = line.rstrip()
    items = line.split(' ')
    num=int(items[2])
    dictory=items[0].split('/')
    motion=dictory[0]
    ranges[num]=motion
count=0
for a in ranges:

    line_hmbd51=''
    line_hmbd51=a+' '+str(count+1)
    count+=1
    with open('/media/hd0/liujiayu/code/TDN-main/utils/ucf101.txt','a')as f:
        f.write(line_hmbd51+'\n')   
    #print(dictory)#['', 'media', 'hd0', 'datasets', 'hmdb51', 'videos', 'fencing', 'Zorro_(Funny_Fight_Scene)_fencing_u_cm_np2_fr_bad_1.avi']
    # if(items[1]=='1')or(items[1]=='2')or(items[1]=='3')or(items[1]=='4')or (items[1]=='5')or (items[1]=='6')or(items[1]=='7'):
    #     with open('/media/hd0/liujiayu/code/TDN-main/utils/hmbd51confusionmax.txt','a')as f:
    #         f.write(line+'\n')

