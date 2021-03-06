# -*- coding: utf-8 -*-
"""Created on Thu Aug 15 15:05:56 2019
@author: Taoting
将用labeime标注的json转化成coco格式的json
"""

import json
import cv2
import numpy as np
import os

#用闭包实现计数器
def counter():
    cnt = 0
    def increce():
        nonlocal cnt
        x = cnt
        cnt += 1
        return x
    return increce

def p_images(data,data_coco):
    images=[]
    image={}
    # image是一个{} 而不是[]
    file_name=data['imagePath'].split('\\')[-1]
    #print(file_name)
    image['file_name']=file_name
    #image['id']=int(file_name.split('.')[0])
    image['id']=1
    image['height']=data['imageHeight']
    image['width']=data['imageWidth']
    img=None
    images.append(image)
    data_coco['images']=images
    return file_name
  
#用一个coco格式的json做参考
def modify_categories():
    ref_json_path = '2011_000003.json'
    data=json.load(open(ref_json_path))
    modified_categories = []
    catNms=['person','bicycle','car','motorcycle','truck','bus','bottle']#根据你的数据修改
    for i,cat in enumerate(data['categories']):

        if cat['name'] in catNms:
            modified_categories.append(cat)
        else:
            pass
    return modified_categories,data['info'],data['licenses']
   
    
def p_annotation(data,data_coco,cnt):  
    # annotations
    annotations=[]

    for i in range(len(data['shapes'])):
        annotation={}
        annotation['segmentation']=[list(np.asarray(data['shapes'][i]['points']).flatten())]   # data['shapes'][0]['points']
        annotation['iscrowd']=0
        annotation['image_id']=data_coco['images'][0]['id']
        #找出标注点中的外接矩形的四个点
        x = annotation['segmentation'][0][::2]#奇数个是x的坐标
        y = annotation['segmentation'][0][1::2]#偶数个是y的坐标
        #print(x,y)
        x_left = min(x)-1#往外扩展1个像素，也可以不扩展
        y_left = min(y)-1
        w = max(x) - min(x)+1
        h = max(y) - min(y)+1
        annotation['bbox']=[x_left,y_left,w,h] # [左上角x,y以及宽和高]
        cat_list_dict = [cat for cat in data_coco['categories'] if cat['name'] == data['shapes'][i]['label'].split('_')[0]]#注意这里是跟标注时填类别的方式有关
        print(cat_list_dict)
        annotation['category_id']=cat_list_dict[0]['id']
        annotation['id'] = cnt() # 第一个对象 这个ID也不能重复，如果下一张图，id不能取1，需从1 开始往下取
        #print('cnt', annotation['id'])
        #print('annotation',annotation)
        annotations.append(annotation)
        #print('annotations',annotations)

    data_coco['annotations']=annotations
    #print(data_coco['annotations'])
    #return data_coco

def Labelme2coco(json_path,cnt):
    with open(json_path,'r') as fp:
        data = json.load(fp)  # 加载json文件
        data_coco={}
        # images
        file_name = p_images(data,data_coco)
        # categories
        modified_categories, info, p_license = modify_categories()
        data_coco['categories'] = modified_categories
        #print(data_coco['categories'])
        # info
        data_coco['info'] = info
        # license
        data_coco['license'] = p_license
        # annotations        
        p_annotation(data,data_coco,cnt)
        #print(data_coco['annotations'])
        return data_coco,file_name

if __name__ == '__main__':
    root_dir = 'lableme-format'
    json_list = os.listdir(root_dir)
    cnt = counter()
    for json_path in json_list:
        if json_path.split('.')[-1] == 'json':
            data_coco,file_name = Labelme2coco(os.path.join(root_dir,json_path),cnt)
            # 保存json文件
            json.dump(data_coco,open('%s_coco.json' % file_name.split('.')[0],'w'),indent=4)
    
