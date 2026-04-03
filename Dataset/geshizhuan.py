from moviepy.editor import VideoFileClip
import os
for root, dirs, files in os.walk(r"/media/hd0/liujiayu/data/20bn-something-something-v2"):
    for file in files:
        if ('.webm' in file)or ('.mkv' in file):
            path = os.path.join(root, file)
            #print(path)
            #print(root)
            input_file = path
            basename=os.path.splitext(file)[0]
            basename=basename+'.mp4'
            trys='/media/hd0/liujiayu/data/somethingv2-mp4'
            output_file=os.path.join(trys, basename)
            clip = VideoFileClip(input_file)
            clip.write_videofile(output_file, codec="png")
            print(f"已成功将 {input_file} 转换为 {output_file}")
            if(os.path.exists(output_file)):
                os.remove(input_file)
                print(f"{input_file} 已成功删除")



#                 print(path)
# # 指定输入文件的路径（.webm格式）
# input_file = "input_video.webm"

# # 指定输出文件的路径（.avi格式）
# output_file = "output_video.avi"

# # 加载视频文件
# clip = VideoFileClip(input_file)

# # 将视频保存为 .avi 格式
# clip.write_videofile(output_file, codec="png")

# print(f"已成功将 {input_file} 转换为 {output_file}")
