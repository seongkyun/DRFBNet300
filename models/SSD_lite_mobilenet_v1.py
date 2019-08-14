import torch
import torch.nn as nn

from collections import namedtuple
import functools

import torch.nn.functional as F
from torch.autograd import Variable

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from layers import *

def conv_bn(inp,oup,stride):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
            nn.Conv2d(inp,inp, kernel_size=3, stride=stride, padding=1,groups = inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
    )

def MobileNet():
    layers = []
    layers += [conv_bn(3, 32, 2)]
    layers += [conv_dw(32, 64, 1)]
    layers += [conv_dw(64, 128, 2)]
    layers += [conv_dw(128, 128, 1)]
    layers += [conv_dw(128, 256, 2)]
    layers += [conv_dw(256, 256, 1)]
    layers += [conv_dw(256, 512, 2)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 512, 1)]
    layers += [conv_dw(512, 1024, 2)]
    layers += [conv_dw(1024, 1024, 1)]

    return layers

class SSDLite(nn.Module):
    """Single Shot Multibox Architecture for embeded system
    See: https://arxiv.org/pdf/1512.02325.pdf & 
    https://arxiv.org/pdf/1801.04381.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, phase, size, base, extras, head, feature_layer, num_classes):
        super(SSDLite, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        # SSD network
        self.base = nn.ModuleList(base)
        self.Norm = L2Norm(feature_layer[1][0], 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        self.feature_layer = feature_layer[0]

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources = list()
        loc = list()
        conf = list()

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if len(sources) == 0:
                    s = self.Norm(x)
                    sources.append(s)
                else:
                    sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            sources.append(x)
            # if k % 2 == 1:
            #     sources.append(x)

        #if phase == 'feature':
        #    return sources

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

def add_extras(base, feature_layer, mbox, num_classes):
    extra_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None
    for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
        if layer == 'S':
            extra_layers += [ _conv_dw(in_channels, depth, stride=2, padding=1, expand_ratio=1) ]
            in_channels = depth
        elif layer == '':
            extra_layers += [ _conv_dw(in_channels, depth, stride=1, expand_ratio=1) ]
            in_channels = depth
        else:
            in_channels = depth
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
    return base, extra_layers, (loc_layers, conf_layers)

# based on the implementation in https://github.com/tensorflow/models/blob/master/research/object_detection/models/feature_map_generators.py#L213
# when the expand_ratio is 1, the implemetation is nearly same. Since the shape is always change, I do not add the shortcut as what mobilenetv2 did.
def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

def build_ssd_lite(phase, size, base, feature_layer, mbox, num_classes):
    base_, extras_, head_ = add_extras(base, feature_layer, mbox, num_classes)
    return SSDLite(phase, size, base_, extras_, head_, feature_layer, num_classes)

ASPECT_RATIOS = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
number_box= [2*len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for aspect_ratios in ASPECT_RATIOS]  

def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Currently SSD300 models are supported!")
        return
    base = MobileNet()
    FEATURE_LAYER = [[11, 13, 'S', 'S', 'S', 'S'], [512, 1024, 512, 256, 256, 128]]

    model = build_ssd_lite(phase=phase, size=size, base=base, feature_layer=FEATURE_LAYER, mbox=number_box, num_classes=num_classes)

    return model

def test(device=None):
    if device == "cpu":
        print('CPU mode')
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    net = build_net('train', 300, 21).to(device)
    print(net)

    from torchsummary import summary
    summary(net, input_size=(3, 300, 300), device=str(device))

    inputs = torch.randn(32, 3, 300, 300).to(device)

    start.record()
    out = net(inputs)
    end.record()

    torch.cuda.synchronize()
    print('32 Batch Relative inf time: {:.2f} ms'.format(start.elapsed_time(end)))
    print('coords output size: ', out[0].size())
    print('class output size: ', out[1].size())

#test("cpu")
#test()

'''
Total params: 7,844,832
Trainable params: 7,844,832
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.03
Forward/backward pass size (MB): 214.05
Params size (MB): 29.93
Estimated Total Size (MB): 245.00
----------------------------------------------------------------
Relative inf time: 78.43 ms
coords output size:  torch.Size([32, 2990, 4])
class output size:  torch.Size([32, 2990, 21])
'''