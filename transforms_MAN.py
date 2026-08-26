import torch
from PIL import Image
import numpy as np
import cv2
import random
from torchvision.transforms import functional


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
    return inner_area


class Transforms(object):
    def __init__(self, scale, crop, stride, gamma, dataset):
        self.scale = scale  # (0.8,1.2)
        self.crop = crop  # (400,400)
        self.stride = stride  # 1
        self.gamma = gamma  # (0.5,1.5)
        self.dataset = dataset  # 'train'

    def __call__(self, image, keypoints, attention):  # density
        # random resize
        height, width = image.size[1], image.size[0]
        if self.dataset == 'train':
            if height < width:
                short = height
            else:
                short = width
            if short < 672:  # 【】原512，现在cropsize为512，需要保证随机缩小0.8倍的时可行，设为672
                scale = 672 / short  # 【】原512
                height = round(height * scale)  # 返回浮点数x的四舍五入值。
                width = round(width * scale)
                image = image.resize((width, height), Image.BILINEAR)
                if len(keypoints) > 0:
                    keypoints = keypoints * scale
                # density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
                # attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)

        scale = random.uniform(self.scale[0], self.scale[1])  # random.uniform(参数1，参数2) 返回参数1和参数2之间的任意值
        height = round(height * scale)
        width = round(width * scale)
        image = image.resize((width, height), Image.BILINEAR)
        if len(keypoints) > 0:
            keypoints = keypoints * scale
        # density = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR) / scale / scale
        # attention = cv2.resize(attention, (width, height), interpolation=cv2.INTER_LINEAR)

        # random crop
        h, w = self.crop[0], self.crop[1]  # 高、宽
        dh = random.randint(0, height - h)  # 上
        dw = random.randint(0, width - w)  # 左
        image = image.crop((dw, dh, dw + w, dh + h))  # (左,上,右,下)
        # 无需判断train，因为train阶段才会进入这个类（下面是处理点的裁剪）
        if len(keypoints) > 0:  # 判断是否>0
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(dw, dh, dw + w, dh + h, bbox)  # (左,上,左+w,上+h)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)
            target = ratio[mask]  # target是类别标签
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [dw, dh]  # (左,上) change coodinate
        else:
            target = np.array([])

        # density = density[dh:dh + h, dw:dw + w]
        # attention = attention[dh:dh + h, dw:dw + w]

        # random flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 图像的翻转
            if len(keypoints) > 0:
                keypoints[:, 0] = w - keypoints[:, 0]
            # density = density[:, ::-1]
            # attention = attention[:, ::-1]

        # random gamma
        if random.random() < 0.3:
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            image = functional.adjust_gamma(image, gamma)  # 对一张图片进行gamma校正,返回：gamma校正的图片

        # random to gray
        if self.dataset == 'train':
            if random.random() < 0.1:
                image = functional.to_grayscale(image, num_output_channels=3)  # 作用：将图像转换为灰度图像

        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # 这里是保证能被步长整除，但是增量和MAN的步长都为1，且点标注不方便这样处理，所以注释
        # density = cv2.resize(density, (density.shape[1] // self.stride, density.shape[0] // self.stride),
        #                      interpolation=cv2.INTER_LINEAR) * self.stride * self.stride
        # attention = cv2.resize(attention, (attention.shape[1] // self.stride, attention.shape[0] // self.stride),
        #                        interpolation=cv2.INTER_LINEAR)

        # attention[attention > 0.0001] = 1
        # attention = attention.astype(np.float32, copy=False)
        # 【】这一步是[a,b]==>[1,a,b]
        # density = np.reshape(density, [1, density.shape[0], density.shape[1]])
        # attention = np.reshape(attention, [1, attention.shape[0], attention.shape[1]])

        # 加上torch.from_numpy等操作，来自MAN
        return image, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(target.copy()).float(), short
