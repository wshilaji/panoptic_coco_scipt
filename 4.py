# -*- coding: utf-8 -*-
"""
@author: Taoting
coco->labelme
"""

import json
import cv2
import numpy as np
import os

def reference_labelme_json():
    ref_json_path = 'DJI_0001.json'
    data=json.load(open(ref_json_path))    
    return data
    
def labelme_shapes(data,data_ref):
    shapes = []
    label_num = {'person':0,'bicycle':0,'car':0,'motorcycle':0,'bus':0,'train':0,'truck':0,'bottle':0}
    for ann in data['annotations']:
        shape = {}
        class_name = [i['name'] for i in data['categories'] if i['id'] == ann['category_id']]
        print('class_name[0]:%s',class_name[0])

        label_num[class_name[0]] += 1
        print(label_num)
        shape['label'] = class_name[0] + '_' + str(label_num[class_name[0]])
        # shape['label'] = class_name[0]
        # shape['label'] = class_name   ////label :[person]  不符合格式 必须[0]才是文本否则是列表
        shape['line_color'] = data_ref['shapes'][0]['line_color']
        shape['fill_color'] = data_ref['shapes'][0]['fill_color']
        
        shape['points'] = []
        # ~ print(ann['segmentation'])
        if not type(ann['segmentation']) == list:
            continue
        else:
            x = ann['segmentation'][0][::2]
            y = ann['segmentation'][0][1::2]

            for j in range(len(x)):
                shape['points'].append([x[j], y[j]])
            
            shape['shape_type'] =  data_ref['shapes'][0]['shape_type']
            shape['flags'] = data_ref['shapes'][0]['flags']
            shapes.append(shape)
    return shapes
        

def Coco2labelme(json_path,data_ref):
    with open(json_path,'r') as fp:
        data = json.load(fp)
        data_labelme={}
        data_labelme['version'] = data_ref['version']
        data_labelme['flags'] = data_ref['flags']
        
        data_labelme['shapes'] = labelme_shapes(data,data_ref)
        
        data_labelme['lineColor'] = data_ref['lineColor']
        data_labelme['fillColor'] = data_ref['fillColor']
        data_labelme['imagePath'] = data['images'][0]['file_name']
        
        data_labelme['imageData'] = None
        # ~ data_labelme['imageData'] = data_ref['imageData']
        
        data_labelme['imageHeight'] = data['images'][0]['height']
        data_labelme['imageWidth'] = data['images'][0]['width']

        return data_labelme

if __name__ == '__main__':
    root_dir = 'coco-format'
    json_list = os.listdir(root_dir)
    print(json_list)
    data_ref = reference_labelme_json()

    for json_path in json_list:
        if json_path.split('.')[-1] == 'json':
            print('cur file:： ', json_path)
            data_labelme= Coco2labelme(os.path.join(root_dir,json_path), data_ref)
            file_name = data_labelme['imagePath']
            json.dump(data_labelme,open('%s_cocotolabelme.json' % os.path.basename(file_name).split('.')[0],'w'),indent=4)
