# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:48:12 2019

@author: Taoting
"""
from __future__ import print_function
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
#from coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annFile='/home/dys/dl/labelme/examples/instance_segmentation/labelme-coco-api/JPEGImages/2011_000003.json'
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())

nms=[cat['name'] for cat in cats]
nms = set([cat['supercategory'] for cat in cats])

imgIds = coco.getImgIds()
print(imgIds)
img = coco.loadImgs(imgIds[0])[0]
print(img)

I = io.imread('%s'%(img['file_name']))
plt.axis('off')
plt.imshow(I)
plt.show()

catIds=[]
for ann in coco.dataset['annotations']:
    if ann['image_id']==imgIds[0]:  
        catIds.append(ann['category_id'])


plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()


