import random

import pandas as pd
import  os
import  re
import  glob
import  time


def remove_spaces_from_directories(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in dirs:
            if ' ' in name:
                original_path = os.path.join(root, name)
                new_path = os.path.join(root, name.replace(' ', ''))
                os.rename(original_path, new_path)
                print(f"重命名：{original_path} -> {new_path}")
# remove_spaces_from_directories(ValDatasetDir)


def kinets():
    list_index=pd.read_csv("/media/hd0/liujiayu/code/TDN-main/utils/category.csv",header=None).iloc[:,0].to_list()
    print(list_index)
    csv_file="/Users/bighu/Downloads/raw-part/compress/val_256.csv"
    val256=pd.read_csv(csv_file,header=None,sep=',')
    # print(val256)
    i,j=val256.iloc[:,1],val256.iloc[:,2]


    list1=val256.iloc[:,0].to_list()
    list2=val256.iloc[:,1].to_list()
    list3=[]
    for i,j in zip(list1,list2):

        # search_pattern = os.path.join('/Users/bighu/Downloads/raw-part/compress/train_256/',i.replace(" ","_"), j+'_[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]' + '.*')
        search_pattern = os.path.join('/Users/bighu/Downloads/raw-part/compress/val_256/', i.replace(" ", "_"),j+ '.*')
        # print(search_pattern)
        matching_files = glob.glob(search_pattern)
        # print(matching_files)

        if len(matching_files)!=0:
            list3.append([matching_files[0],list_index.index(i.replace(" ","_"))])



    df=pd.DataFrame(data=list3)
    # for i,j in zip(list1,list2):
        # df.loc[len(df.index)]=[ValDatasetDir+i+'/'+j,j]
        # print(i,j)
    df=df.sample(n=100,random_state=int(time.time()))
    df.to_csv("./new.csv",index=None,header=None,sep=' ')
    # df.to_csv("./new.csv",index=None,header=None,sep=' ')



def createCsv_hmdb51():
    root_dir="/media/hd1/datasets/hmdb51/videos"
    # data=pd.read_csv("/Users/bighu/Downloads/testTrainMulti_7030_splits/brush_hair_test_split1.txt",header=None,sep=' ')
    data=pd.read_csv("/Users/bighu/Downloads/testTrainMulti_7030_splits/brush_hair_test_split2.txt",header=None,sep=' ')
    list1=[ [os.path.join(root_dir,dir),index]   for dir,index in zip(data.iloc[:,0].to_list(),data.iloc[:,1].to_list())]
    data1=pd.DataFrame(list1)
    data1.to_csv("./hmdb51.csv",header=None,index=None,sep=' ')


def getDirList(dir):
    root_dir = "/media/hd1/datasets/hmdb51/videos"
    csvFile = {}
    dataFile=[]

    files = os.listdir(dir)
    # files.sort()
    # print(files)
    for file in files:
        if file.find("split1") != -1:
            continue
        categroy = file.split("_test_")[0]
        print(categroy)
        # print(os.path.join(dir, file))
        data=pd.read_csv(os.path.join(dir, file),header=None,sep=' ')
        # print(data.iloc[:,1].to_list())
        # print(categroy)
        if categroy not in csvFile:
            csvFile[categroy] = []
        for x, y in zip(data.iloc[:, 0].to_list(), data.iloc[:, 1].to_list()):
                csvFile[categroy].append([os.path.join(os.path.join(root_dir, categroy), x), y,categroy])
    i=0
    for x in csvFile.values():
        for y in x:
            # print(y)
            i+=1
            dataFile.append(y)
    # print(i)

    categroyList=list(csvFile.keys())
    # print(dataFile)
    train_data=[ [x,categroyList.index(z)]  for x,y,z in dataFile if y==1  ]
    test_data=[ [x,categroyList.index(z)]  for x,y,z in dataFile  if y==2 ]
    val_data=[ [x,categroyList.index(z)]  for x,y,z in dataFile  if y==0 ]
    print(len(train_data))
    print(len(test_data))
    print(len(val_data))
    df = pd.DataFrame(train_data)

    df=df.drop_duplicates()
    df=df.drop([1,2,3])
    df.info()

    df.to_csv("./train.csv",sep=' ',header=None,index=None)
    df = pd.DataFrame(test_data)

    df=df.drop_duplicates()
    df.info()
    df.to_csv("./val.csv",sep=' ',header=None,index=None)
    # pass

def getCsv_hmdb51():
    db51_dir="/media/hd0/liujiayu/code/TDN-main/utils"
    labelDir="/media/hd0/datasets/hmdb51/labels"
    dataDir="/media/hd0/datasets/hmdb51/videos"
    categoryList=pd.read_csv("./category.csv",header=None).iloc[:,0].to_list()

    files_and_directories=os.listdir(labelDir)
    files = [file for file in files_and_directories if os.path.isfile(os.path.join(labelDir, file))]
    train_list=[]
    test_list=[]
    for file in files:
        if file.find('split1')!= -1:
            category=file.split("_test_")[0]
            df=pd.read_csv(os.path.join(labelDir,file),header=None,sep=' ')
            # print(len(df.iloc[:,0].to_list()))
            for a in [[os.path.join(dataDir,category+"/"+x),categoryList.index(category)] for x,y in zip(df.iloc[:,0].to_list(),df.iloc[:,1].to_list()) if y==1]:
                train_list.append(a)
            for a in [[os.path.join(dataDir,category+"/"+x),categoryList.index(category)] for x,y in zip(df.iloc[:,0].to_list(),df.iloc[:,1].to_list()) if y==2]:
                test_list.append(a)

    print(len(train_list))
    print(len(test_list))
    df = pd.DataFrame(train_list)
    df.info()
    df.to_csv(os.path.join(db51_dir,"train.csv"), sep=' ', header=None, index=None)
    df = pd.DataFrame(test_list)
    df.info()
    df.to_csv(os.path.join(db51_dir,"val.csv"), sep=' ', header=None, index=None)
    pass

# getDirList("/raid5/huwenfeng/dataset/hmdb51/labels")
# createCsv_hmdb51()

getCsv_hmdb51()
