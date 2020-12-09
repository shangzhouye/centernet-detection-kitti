import os
import cv2
import math
import random
import torch.utils.data as data
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
import gc; gc.enable() # memory is tight
import torch

import torch.utils.data as data


dtype = "float32"

## HeatMap Genrating Functions

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1) 
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right] 
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right] 
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def reverse_res_to_bbox(res):
    down_ratio = res['input'].shape[1] / res['hm'].shape[1]
    output_h, output_w =  res['hm'].shape[0],  res['hm'].shape[1]
    num_objs = np.sum(res['reg_mask'])
    bbox = np.zeros((num_objs, 4), dtype = 'float32' )
    ct = res['reg'][:num_objs]
    ct[:, 0] += res['ind'][:num_objs] % output_w
    ct[:,1] += res['ind'][:num_objs] // output_w
    h, w = res['wh'][:num_objs,1],  res['wh'][:num_objs,0]
    bbox[:,0] = (ct[:,0] * 2 - w) /2
    bbox[:,2] = (ct[:,0] * 2 + w) /2
    bbox[:,1] = (ct[:,1] * 2 - h) /2
    bbox[:,3] = (ct[:,1] * 2 + h) /2
    bbox *= down_ratio
    return bbox

class ctDataset(data.Dataset):

    def __init__(self):
        # img_dir = 'kitti dataset/'
        # ../input/kitti_single/training/label_2/
        train_label_dir = '/home/shangzhouye/Documents/kitti_object_2d/data_object_label_2/training/label_2/'
        # label_test_image_dir = os.path.join(os.getcwd(), 'streets\\test\\labels\\')

        train_image_dir = '/home/shangzhouye/Documents/kitti_object_2d/data_object_image_2/training/image_2/'
        # test_image_dir = os.path.join(os.getcwd(), 'streets\\test\\images\\')

        train_calib_dir = '/home/shangzhouye/Documents/kitti_object_2d/data_object_calib/training/calib/'

        images =  [(train_image_dir+f) for f in listdir(train_image_dir) if isfile(join(train_image_dir, f))]
        labels = [(train_label_dir+f) for f in listdir(train_label_dir) if isfile(join(train_label_dir, f))]
        calibs = [(train_calib_dir+f) for f in listdir(train_calib_dir) if isfile(join(train_calib_dir, f))]

        df = pd.DataFrame(np.column_stack([images, labels, calibs]), columns=['images', 'labels', 'calibs'])

        df1 = df.sort_values(by='images')['images'].reset_index()
        # df1 = df.sort_values(by='a')['a']
        df2 = df.sort_values(by='labels')['labels'].reset_index()
        # df2 = df.sort_values(by='b')['b']
        df3 = df.sort_values(by='calibs')['calibs'].reset_index()

        df['images'] = df1['images']
        df['labels'] = df2['labels']
        df['calibs'] = df3['calibs']
        del df1, df2, df3
        self.df_in_list_ = (df).values.tolist()
        # print("Number of samples: ", len(self.df_in_list_))


    def __len__(self):
        return len(self.df_in_list_)

    def __getitem__(self, index):
        image_path, label_path, cali_path = self.df_in_list_[index]
        # get the image (375, 1242, 3)
        img = cv2.imread(image_path) 
        default_resolution = [375, 1242]

        # height, width = img.shape[0], img.shape[1]  
        # print("Before pre-reshape: ", height, width)

        # reshape image to default size (some samples have slightly different size)
        img = cv2.resize(img,(default_resolution[1], default_resolution[0]))

        # get the labels
        with open(label_path) as f:
            content = f.readlines()
        content = [x.split() for x in content]
        # print(content)

        # transform to 512 * 512
        height, width = img.shape[0], img.shape[1]  
        # print("After pre-reshape: ", height, width)
        input_h, input_w = 512, 512 
        inp = cv2.resize(img,(input_w, input_h))
        scale_h, scale_w = input_h/height, input_w/width
        # print(scale_h, scale_w)

        
        inp = (inp.astype(np.float32) / 255.)  
        inp = inp.transpose(2, 0, 1) 

        
        max_objs = 128
        down_ratio = 4 
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = 1
        draw_gaussian = draw_umich_gaussian
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)   
        reg_mask = np.zeros((max_objs), dtype=np.uint8) 
        wh = np.zeros((max_objs, 2), dtype=np.float32)
        reg = np.zeros((max_objs, 2), dtype=np.float32) 
        ind = np.zeros((max_objs), dtype=np.int64) 
        
        count = 0
        for c in content:
            if (c[0] == "Car"):
                bbox = np.array(c[4:8], dtype = "float32")
                bbox[1::2] *= scale_h
                bbox[0::2] *= scale_w
                bbox = bbox/down_ratio
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))  
                    radius = max(0, int(radius))
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) 
                    ct_int = ct.astype(np.int32) 
                    draw_gaussian(hm[0], ct_int, radius)
                    wh[count] = 1. * w, 1. * h
                    ind[count] = ct_int[1] * output_w + ct_int[0]  
                    reg[count] = ct - ct_int
                    reg_mask[count] = 1
                    count = count + 1

        res = {'image': img, \
                'input': inp, \
                'hm': hm, \
                'reg_mask': reg_mask, \
                'ind': ind, \
                'wh': wh, \
                'reg':reg, \
                'index': index}

        return res

if __name__ == "__main__":
    im_idx = 1000
    my_dataset = ctDataset()
    res = my_dataset.__getitem__(im_idx)

    img = res['image']
    inp = res['input'].transpose(1,2,0)
    hm = res['hm']
    print(inp.dtype)

    plt.title("Original Image")
    plt.imshow(img)
    plt.show()


    plt.title("Ground Truth Heat Map")
    im = plt.imshow(hm[0])
    plt.colorbar(im)
    plt.show()


    bbox = reverse_res_to_bbox(res)
    for b in bbox:
        cv2.rectangle(inp, (b[0], b[1]), (b[2], b[3]), (0,0,1), 2) 

    plt.title("Calculated bounding box positions")
    plt.imshow(inp)
    plt.show()