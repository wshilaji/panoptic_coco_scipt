# -*- coding:utf-8 -*-
"""
@author: Taoting
"""
import json
import matplotlib.pyplot as plt
import skimage.io as io
from labelme import utils

def main():
    json_path = 'DJI_0001.json'
    data = json.load(open(json_path))
    #img = io.imread('%s/%s'%('./PATH/TO/IMAGE',data['imagePath']))
    img = io.imread('%s'%(data['imagePath']))
    lab, lab_names = utils.labelme_shapes_to_label(img.shape, data['shapes']) 
    captions = ['%d: %s' % (l, name) for l, name in enumerate(lab_names)]
    lab_ok = utils.draw_label(lab, img, captions)

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lab_ok)
    plt.show()


if __name__ == '__main__':
    main()
