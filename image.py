import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2


def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    target_1 = cv2.resize(target, (target.shape[1] // 8 * 8, target.shape[0] // 8 * 8), interpolation=cv2.INTER_CUBIC)
    if target_1.sum() != 0:
        ration = target.sum() / target_1.sum()
        target_1 = target_1 * ration
    return img, target_1
