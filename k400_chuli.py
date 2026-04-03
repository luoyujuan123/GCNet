import decord
import os
import pdb

def recursive_listdir(path):
    flag=1
    global index
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)

        if os.path.isfile(file_path):
            #print(file_path)
            try:
                video_list = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=1)
            except Exception as e:
                flag=0
                print(flag)
                pass
            if (flag==1):
                with open("/media/hd0/liujiayu/code/TDN-main/utils/kval256.txt","a") as f:  # 打开文件
                    str1=file_path+' '+str(index)
                    f.write(str1+'\n')  # 读取文件
            flag=1                
            continue
        elif os.path.isdir(file_path):
            recursive_listdir(file_path)
            index+=1
index=0
recursive_listdir('/media/hd0/datasets/kinetics/val_256')
print("finish all!!!")
