import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import sys
from torch.nn.modules.upsampling import Upsample
from models.nn_modules import Conv, C2f, SPPF, Concat, Detect
from ultralytics import  YOLO

sys.path.append('../')

from models.yolo_layer import YoloLayer
from models.darknet_utils import parse_cfg, print_cfg, load_fc, load_conv_bn, load_conv
from utils.torch_utils import to_cpu


def reorg(x):
    stride = 2
    assert (x.data.dim() == 4)
    B = x.data.size(0)
    C = x.data.size(1)
    H = x.data.size(2)
    W = x.data.size(3)
    assert (H % stride == 0)
    assert (W % stride == 0)
    ws = stride
    hs = stride
    x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
    x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
    x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
    x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
    return x

#####################
### NEW CLASS FOR YOLOv8
#####################

class ComplexYOLOv8(nn.Module):
    def __int__(self, cfgfile, use_giou_loss):
        super().__int__()
        self.use_giou_loss = use_giou_loss
        # self.blocks = parse_cfg(cfgfile)
        # self.width = int(self.blocks[0]['width'])
        # self.height = int(self.blocks[0]['height'])

        self.models = self.create_network()  # merge conv, bn,leaky
        self.yolo_layers = [layer for layer in self.models if layer.__class__.__name__ == 'YoloLayer']

        self.loss = self.models[len(self.models) - 1]

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0
        self.conv_1 = Conv(3, 16, 3, 2)

    def forward(self, x):
        """ Forward step for yolov8 """
        x_1 = self.conv_1.forward(x)
        x_1 = self.relu(x_1)
        x_1 = self.relu(self.conv_2.forward(x_1))
        x_1 = self.relu(self.c2f_1.forward(x_1))
        x_1 = self.relu(self.conv_3.forward(x_1))
        x_1 = self.relu(self.c2f_2.forward(x_1))

        x_2 = self.relu(self.conv_4.forward(x_1))
        x_2 = self.relu(self.c2f_3.forward(x_2))

        x_3 = self.relu(self.conv_5.forward(x_2))
        x_3 = self.relu(self.c2f_4.forward(x_3))
        x_3 = self.relu(self.sppf_1.forward(x_3))

        x_4 = self.relu(self.upsample_1.forward(x_3))

        x_5 = self.concat_1.forward((x_2, x_4))

        x_6 = self.relu(self.c2f_5.forward(x_5))
        x_7 = self.upsample_2.forward(x_6)

        x_8 = self.concat_2.forward((x_1, x_7))

        x_9 = self.c2f_6.forward(x_8)
        x_9 = self.conv_6.forward(x_9)

        x_10 = self.concat_3.forward((x_9, x_6))

        x_10 = self.c2f_7.forward(x_10)
        x_10 = self.conv_7.forward(x_10)

        x_11 = self.concat_4.forward((x_10, x_3))
        x_12 = self.c2f_8.forward(x_11)
        x_12 = self.conv_8.forward(x_12)
        x_12 = self.relu.forward(x_12)
        # x_12 = self.detect.forward(x_12)

        return x_12

    def create_network(self, blocks):
        models = nn.ModuleList()
        model = nn.Sequential()
        model.add_module("first conv", nn.Conv2d(3, 32,3, 2))
        models.append(model)
        print("This is the model created", models)
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

#########
class old_ComplexYOLOv8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = Conv(3, 16, 3, 2)
        self.conv_2 = Conv(16, 32, 3, 2)
        self.c2f_1 = C2f(32, 32, 1, True)
        self.conv_3 = Conv(32, 64, 3, 2)
        self.c2f_2 = C2f(64, 64, 2, True)
        self.conv_4 = Conv(64, 128, 3, 2)
        self.c2f_3 = C2f(128, 128, 2, True)
        self.conv_5 = Conv(128, 256, 3, 2)
        self.c2f_4 = C2f(256, 256, 1, True)
        self.sppf_1 = SPPF(256, 256, 5)
        self.upsample_1 = Upsample(None, 2, 'nearest')
        self.concat_1 = Concat()
        self.c2f_5 = C2f(384, 128, 1)
        self.upsample_2 = Upsample(None, 2, 'nearest')
        self.concat_2 = Concat()
        self.c2f_6 = C2f(192, 64, 1)
        self.conv_6 = Conv(64, 64, 3, 2)
        self.concat_3 = Concat()
        self.c2f_7 = C2f(192, 128, 1)
        self.conv_7 = Conv(128, 128, 3, 2)
        self.concat_4 = Concat()
        self.c2f_8 = C2f(384, 256, 1)
        # self.detect = Detect(8, (256, 128, 256))
        self.relu = nn.ReLU(inplace=True)
        self.conv_8 = Conv(256, 75, 1, 1)

    def forward(self, x):
        """ Forward step for yolov8 """
        x_1 = self.conv_1.forward(x)
        x_1 = self.relu(x_1)
        x_1 = self.relu(self.conv_2.forward(x_1))
        x_1 = self.relu(self.c2f_1.forward(x_1))
        x_1 = self.relu(self.conv_3.forward(x_1))
        x_1 = self.relu(self.c2f_2.forward(x_1))

        x_2 = self.relu(self.conv_4.forward(x_1))
        x_2 = self.relu(self.c2f_3.forward(x_2))

        x_3 = self.relu(self.conv_5.forward(x_2))
        x_3 = self.relu(self.c2f_4.forward(x_3))
        x_3 = self.relu(self.sppf_1.forward(x_3))

        x_4 = self.relu(self.upsample_1.forward(x_3))

        x_5 = self.concat_1.forward((x_2, x_4))

        x_6 = self.relu(self.c2f_5.forward(x_5))
        x_7 = self.upsample_2.forward(x_6)

        x_8 = self.concat_2.forward((x_1, x_7))

        x_9 = self.c2f_6.forward(x_8)
        x_9 = self.conv_6.forward(x_9)

        x_10 = self.concat_3.forward((x_9, x_6))

        x_10 = self.c2f_7.forward(x_10)
        x_10 = self.conv_7.forward(x_10)

        x_11 = self.concat_4.forward((x_10, x_3))
        x_12 = self.c2f_8.forward(x_11)
        x_12 = self.conv_8.forward(x_12)
        x_12 = self.relu.forward(x_12)
        # x_12 = self.detect.forward(x_12)

        return x_12