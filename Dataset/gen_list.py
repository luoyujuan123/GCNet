# gen_list.py
import os
import pdb
import datetime
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='data_path of videos, absolute path')
parser.add_argument('outfile', type=str, help='output.txt file')
args = parser.parse_args()

start_time = datetime.datetime.now()
# get label dictionary 
labels = [ i for i in os.listdir(args.path)]
labels.sort()
if '.DS_Store' in labels:
    labels.remove('.DS_Store')
dic = {label:idx for (idx, label) in enumerate(labels)}


# get [video_path, num_of_frames, labels] 
tt = 0
dirss = [i for i in os.listdir(args.path)]
dirss.sort()
# print(dirss)

record = []
dic_cor = 0
for dirs in dirss[:]:
    if dirs == '.DS_Store':
        continue
    print(dic_cor)
    dic_cor += 1
    frames_path = []
    for video in os.listdir(os.path.join(args.path, dirs)):
        if video == '.DS_Store':
            continue
        # print(os.path.join(args.path, dirs, video))
        frames_path = [i for i in os.listdir(os.path.join(args.path, dirs, video))]
        frames_len = len(frames_path) - 2 if ".DS_Store" in frames_path else len(frames_path)-1
        # print(dirs, video) 
        record.append([os.path.join('train_256/',dirs, video), frames_len, dic[dirs]])
        #pdb.set_trace()
        tt += 1
        if tt % 10000 == 0:
            print('record:', tt)
            with open(args.outfile,"a") as f:
                for i in range(len(record)):
                    rec =  str(record[i][0] + ' ' + str(record[i][1]) + ' ' + str(record[i][2]) + '\n')
                    f.write(rec)
                record = []

with open(args.outfile,"a") as f:
    for i in range(len(record)):
        rec =  str(record[i][0] + ' ' + str(record[i][1]) + ' ' + str(record[i][2]) + '\n')
        f.write(rec)

print("Run time:", datetime.datetime.now()-start_time)
