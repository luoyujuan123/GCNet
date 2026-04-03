import os
import cv2
import json
cut_frame = 3  # 多少帧截一次，自己设置就行
save_path = "/home/liujiayu/data/somethingV2/20bn-something-something-v2-frames"

# for root, dirs, files in os.walk(r"/home/liujiayu/data/somethingV2/20bn-something-something-v2"):  # 这里就填文件夹目录就可以了
#     for file in files:
#         # 获取文件路径
#         if ('.webm' in file):
#             path = os.path.join(root, file)
#             print(path)
#             file_name = os.path.basename(path)
#             video = cv2.VideoCapture(path)
#             video_fps = int(video.get(cv2.CAP_PROP_FPS))
#             print(video_fps)
#             current_frame = 0
#             while (True):             
#                 basename=os.path.splitext(file_name)[0]
#                 save=save_path + '/' +basename
#                 if not os.path.exists(save):
#                     os.makedirs(save)
#                 ret, image = video.read()
#                 current_frame = current_frame + 1
#                 if ret is False:
#                     video.release()
#                     break
#                 if current_frame % cut_frame == 0:
#                     # cv2.imwrite(save_path + '/' + file[:-4] + str(current_frame) + '.jpg',
#                     #             image)  # file[:-4]是去掉了".mp4"后缀名，这里我的命名格式是，视频文件名+当前帧数+.jpg，使用imwrite就不能有中文路径和中文文件名
#                     cv2.imencode('.jpg', image)[1].tofile(save_path + '/' +os.path.splitext(file_name)[0]+'/'+ os.path.splitext(file_name)[0] +'_'+ str(current_frame) + '.jpg') #使用imencode就可以整个路径中可以包括中文，文件名也可以是中文
#                     print('正在保存' + file + save_path + '/' +os.path.splitext(file_name)[0]+'/'+ os.path.splitext(file_name)[0] +'_'+ str(current_frame))

with open('/home/liujiayu/code/TDN-main/Dataset/something-something-v2-train.json', 'r') as jsons:
    data = json.load(jsons)
for i in range(len(data)):
    for root, dirs, files in os.walk(r"/home/liujiayu/data/somethingV2/20bn-something-something-v2"):
        for file in files:
            # 获取文件路径
            file_ns=data[i]["id"]+'.webm'
            if (file_ns in file) and(not os.path.exists(save_path+'/'+data[i]["id"])):
                path = os.path.join(root, file)
                print(path)
                file_name = os.path.basename(path)
                video = cv2.VideoCapture(path)
                video_fps = int(video.get(cv2.CAP_PROP_FPS))
                print(video_fps)
                current_frame = 0
                while (True):             
                    basename=os.path.splitext(file_name)[0]
                    save=save_path + '/' +basename
                    if not os.path.exists(save):
                        os.makedirs(save)
                    ret, image = video.read()
                    current_frame = current_frame + 1
                    if ret is False:
                        video.release()
                        break
                    if current_frame % cut_frame == 0:
                        # cv2.imwrite(save_path + '/' + file[:-4] + str(current_frame) + '.jpg',
                        #             image)  # file[:-4]是去掉了".mp4"后缀名，这里我的命名格式是，视频文件名+当前帧数+.jpg，使用imwrite就不能有中文路径和中文文件名
                        cv2.imencode('.jpg', image)[1].tofile(save_path + '/' +os.path.splitext(file_name)[0]+'/'+ os.path.splitext(file_name)[0] +'_'+ str(current_frame) + '.jpg') #使用imencode就可以整个路径中可以包括中文，文件名也可以是中文
                        print('正在保存' + file + save_path + '/' +os.path.splitext(file_name)[0]+'/'+ os.path.splitext(file_name)[0] +'_'+ str(current_frame))
    i=i+1

# for root, dirs, files in os.walk(r"/home/liujiayu/data/somethingV2/20bn-something-something-v2"):  # 这里就填文件夹目录就可以了
#     for file in files:
#         # 获取文件路径
#         if ('9119.webm' in file):
#             path = os.path.join(root, file)
#             print(path)
#             file_name = os.path.basename(path)
#             video = cv2.VideoCapture(path)
#             video_fps = int(video.get(cv2.CAP_PROP_FPS))
#             print(video_fps)
#             current_frame = 0
#             while (True):             
#                 basename=os.path.splitext(file_name)[0]
#                 save=save_path + '/' +basename
#                 if not os.path.exists(save):
#                     os.makedirs(save)
#                 ret, image = video.read()
#                 current_frame = current_frame + 1
#                 if ret is False:
#                     video.release()
#                     break
#                 if current_frame % cut_frame == 0:
#                     # cv2.imwrite(save_path + '/' + file[:-4] + str(current_frame) + '.jpg',
#                     #             image)  # file[:-4]是去掉了".mp4"后缀名，这里我的命名格式是，视频文件名+当前帧数+.jpg，使用imwrite就不能有中文路径和中文文件名
#                     cv2.imencode('.jpg', image)[1].tofile(save_path + '/' +os.path.splitext(file_name)[0]+'/'+ os.path.splitext(file_name)[0] +'_'+ str(current_frame) + '.jpg') #使用imencode就可以整个路径中可以包括中文，文件名也可以是中文
#                     print('正在保存' + file + save_path + '/' +os.path.splitext(file_name)[0]+'/'+ os.path.splitext(file_name)[0] +'_'+ str(current_frame))
