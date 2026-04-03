import os
import time
from pathlib import Path


# 遍历文件夹及其子文件夹中的文件，并存储在一个列表中

# 输入文件夹路径、空文件列表[]

# 返回 文件列表Filelist,包含文件名（完整路径）

# def mp4_gai(dir):
#     for root, dirs, files in os.walk(dir):
#         for file in files:
#             if('.webm' in file) or ('.mkv' in file):
                



def get_filelist(dir, Filelist):
    newDir = dir

    if os.path.isfile(dir):

        Filelist.append(dir)

        # # 若只是要返回文件文，使用这个

        # Filelist.append(os.path.basename(dir))

    elif os.path.isdir(dir):

        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码

            if s == "UCF-101":

             continue

            newDir = os.path.join(dir, s)

            get_filelist(newDir, Filelist)

    return Filelist


# def Video2Mp4(videoPath, outVideoPath):
#     capture = cv2.VideoCapture(videoPath)
#     fps = capture.get(cv2.CAP_PROP_FPS)  # 获取帧率
#     size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     # fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     suc = capture.isOpened()  # 是否成功打开

#     allFrame = []
#     while suc:
#         suc, frame = capture.read()
#         if suc:
#             allFrame.append(frame)
#     capture.release()

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     outVideoPath=str(outVideoPath)
#     videoWriter = cv2.VideoWriter(outVideoPath, fourcc, fps, size)
#     for aFrame in allFrame:
#         videoWriter.write(aFrame)
#     videoWriter.release()

#创造.mp4数据集
# if __name__ == '__main__':
#
#     list = get_filelist('/raid5/liujiayu/TDN-main/dataset_root', [])
#     goal_dir='/raid5/liujiayu/TDN-main/Dataset/101/'
#     print(len(list))
#
#     for e in list:
#         if not os.path.exists(goal_dir+os.path.split(os.path.split(e)[0])[1]):
#             os.makedirs(goal_dir+os.path.split(os.path.split(e)[0])[1])
#
#         newdir=goal_dir+os.path.split(os.path.split(e)[0])[1]+'/'+os.path.basename(e)
#         #print(newdir)
#         path_str = Path(newdir)
#         path_suffix = path_str.with_suffix(".mp4")
#         Video2Mp4(e,path_suffix)
#         print(path_suffix)

        #print(e,os.path.split(os.path.split(e)[0])[1])

#创建txt
if __name__ == '__main__':
    list = get_filelist('/mnt/data/liujiayu/Kinetics-400/raw-part/compress/val_256', [])
    f = open('/home/liujiayu/code/TDN-main/Dataset/vallist.txt', 'r+')
    i=0
    folder_name=' '
    for e in list:
        if(folder_name!=os.path.split(os.path.split(e)[0])[1])and(folder_name!=' '):
            i=i+1
        folder_name=os.path.split(os.path.split(e)[0])[1]
        print(e, os.path.split(os.path.split(e)[0])[1])
        #items=e+' '+os.path.split(os.path.split(e)[0])[1]
        items=e+' '+str(i)
        f.write(items+'\n')
    f.close()
