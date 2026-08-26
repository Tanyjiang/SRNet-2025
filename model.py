import torch.nn as nn
import torch
from torchvision import models
from utils import save_net, load_net


class CSRNet(nn.Module):
    def __init__(self, load_model='', downsample=1, bn=True):
        super(CSRNet, self).__init__()
        self.downsample = downsample
        self.device = torch.device('cuda:0')
        self.bn = bn
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.features = make_layers(self.features_cfg, batch_norm=self.bn, dilation=False)

        self.avp = nn.AdaptiveAvgPool2d((1, 1))
        self.front_cfg = [512, 512, 256, 128]
        self.frontend = make_layers(self.front_cfg, in_channels=512, dilation=True)

        self.load_model = load_model
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        y = self.avp(x)
        y = y.view(y.size(0), -1)
        x = self.frontend(x)
        z = self.avp(x)
        z = z.view(z.size(0), -1)
        return x, y, z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_weights(self):
        if not self.load_model:
            pretrained_model = models.vgg16_bn(pretrained=True)
            self._initialize_weights()
            self.features.load_state_dict(pretrained_model.features[0:32].state_dict())
        else:
            self.load_state_dict(torch.load(self.load_model))
            print(self.load_model, ' loaded!')


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
