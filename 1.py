# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Aug 15 13:58:40 2019

@author: Taoting
"""

import matplotlib.pyplot as plt
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import pylab
import json
from pycocotools.coco import COCO

pylab.rcParams['figure.figsize'] = (8.0, 10.0)


json_file='000000397133.json'

data=json.load(open(json_file,'r'))

data_2={}
data_2['info']=data['info']
data_2['licenses']=data['licenses']
data_2['images']=[data['images'][0]] 

data_2['categories']=data['categories']
annotation=[]

for ann in data['annotations']:
    if ann['image_id']==data_2['images'][0]['id']:
        annotation.append(ann)        

data_2['annotations']=annotation

json.dump(data_2,open('./first.json','w'),indent=4)


