import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image, ImageDraw
import math

def plot_heapmap(heatmap):
    ''' Plot the predicted heatmap

        Args:
            heatmap ([h, w]) - the heatmap output from keypoint estimator
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)
    ax.set_title("Prediction Heatmap")
    fig.tight_layout()
    plt.show()

def generate_gt_data(index, wh_gt, reg_gt):
    ''' Generate GT data as a detection result for testing
    '''
    height_idx = int(index % 128)
    width_idx = int(index / 128)
    # print(height_idx, width_idx)

    hm = torch.zeros((1, 1, 128, 128))
    hm[0, 0, height_idx, width_idx] = 1

    wh = torch.zeros((1, 2, 128, 128))
    wh[0, 0, height_idx, width_idx] = wh_gt[0]
    wh[0, 1, height_idx, width_idx] = wh_gt[1]

    reg = torch.zeros((1, 2, 128, 128))
    reg[0, 0, height_idx, width_idx] = reg_gt[0]
    reg[0, 1, height_idx, width_idx] = reg_gt[1]


    return hm, wh, reg

def gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat