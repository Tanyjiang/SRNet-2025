import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from masksembles.torch import Masksembles2D
from models_MAN.transformer_cosine import TransformerEncoder, TransformerEncoderLayer


__all__ = ['vgg19_trans']
model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

class VGG_Trans(nn.Module):
    def __init__(self, features):
        super(VGG_Trans, self).__init__()
        self.features = features

        self.avp = nn.AdaptiveAvgPool2d((1, 1))

        d_model = 512
        nhead = 2
        num_layers = 4
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_layers, if_norm)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)

        )

        self.mask_index = 2
        self.n = 8
        self.reg_layer_mask = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Masksembles2D(512, self.n, 2.0),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16
        x = self.features(x)
        if self.training:
            x_mask = self.reg_layer_mask(x)
            x_mask = F.interpolate(x_mask, size=(rh, rw), mode='bilinear', align_corners=False)
        else:
            x_mask = None

        y = self.avp(x)
        y = y.view(y.size(0), -1)

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.encoder(x, (h, w))
        x = x.permute(1, 2, 0).view(bs, c, h, w)

        x = F.interpolate(x, size=(rh, rw), mode='bilinear', align_corners=False)
        x = self.reg_layer_0(x)

        z = self.avp(x)
        z = z.view(z.size(0), -1)
        return x, y, z, x_mask

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def vgg19_trans():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Trans(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
