from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(mat_path):

    points = loadmat(mat_path)['annPoints'].astype(np.float32)

    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir-img', default=r'',
                        help='original data directory')
    parser.add_argument('--origin-dir-mat', default=r'',
                        help='original data directory')
    parser.add_argument('--data-dir', default=r'',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048


    for phase in ['test_data']:
        sub_dir = os.path.join(args.origin_dir_img, phase)
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        jpg_list = glob(os.path.join(os.path.join(sub_dir, 'images'), '*jpg'))
        mat_list = []

        mat_list = [os.path.join(args.origin_dir_mat, 'GT_' + os.path.basename(jpg_path).replace('.jpg', '.mat')) for jpg_path in jpg_list]
        for mat_path in mat_list:
            name = os.path.basename(mat_path)
            print(name)

            points = loadmat(mat_path)['image_info'][0][0][0][0][0].astype(np.float32)
            if phase == 'train_data':
                dis = find_dis(points)
                points = np.concatenate((points, dis), axis=1)

            mat_save_path = os.path.join(sub_save_dir, name[3:])

            gd_save_path = mat_save_path.replace('mat', 'npy')
            np.save(gd_save_path, points)


