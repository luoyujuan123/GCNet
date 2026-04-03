import json
import string
import re
def remove(text):
    #text = '''Don't worry, be happy!''' # 'Don\'t worry, be happy'

    punctuation_string = string.punctuation
    for i in punctuation_string:
        text = text.replace(i, '')
    #print(text)
    return text

def remove_brackets(text):
    # 使用正则表达式去除方括号及其内容
    return re.sub(r'\[|\]', '', text)

with open('/home/liujiayu/data/something-something-v2-validation.json', 'r') as file:
    data = json.load(file)
with open('/home/liujiayu/data/something-something-v2-labels.json', 'r') as file_label:
    label = json.load(file_label)
# print(data[0]["id"])
print(label["Holding something next to something"])
# remove(data[0]["template"])
with open("val_some.txt",'a') as f:
   for datas in data:
    str_loca=datas["id"]+'.mp4'
    print(datas["template"])
    str_label=label[remove_brackets(datas["template"])]
    str_one=str_loca+' '+str_label
    f.write(str_one+'\n')

