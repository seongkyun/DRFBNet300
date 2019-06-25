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

Conv = namedtuple('Conv', ['stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't']) # t is the expension factor

V1_CONV_DEFS = [
    Conv(stride=2, depth=32),
    DepthSepConv(stride=1, depth=64),
    DepthSepConv(stride=2, depth=128),
    DepthSepConv(stride=1, depth=128),
    DepthSepConv(stride=2, depth=256),
    DepthSepConv(stride=1, depth=256),
    DepthSepConv(stride=2, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=2, depth=1024),
    DepthSepConv(stride=1, depth=1024)
]

V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=2, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]

class _conv_bn_m(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_bn_m, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _conv_dw_m(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw_m, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)

class _inverted_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_inverted_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.depth = oup
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def mobilenet(conv_defs, depth_multiplier=1.0, min_depth=8):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    layers = []
    in_channels = 3
    for conv_def in conv_defs:
        if isinstance(conv_def, Conv):
            layers += [_conv_bn_m(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, DepthSepConv):
            layers += [_conv_dw_m(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, InvertedResidual):
          for n in range(conv_def.num):
            stride = conv_def.stride if n == 0 else 1
            layers += [_inverted_residual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t)]
            in_channels = depth(conv_def.depth)
    return layers

def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

mobilenet_v1 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v1_075 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.25)

mobilenet_v2 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v2_075 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.25)

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
    base_, extras_, head_ = add_extras(base(), feature_layer, mbox, num_classes)
    return SSDLite(phase, size, base_, extras_, head_, feature_layer, num_classes)

ASPECT_RATIOS = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
number_box= [2*len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for aspect_ratios in ASPECT_RATIOS]  

def build_net(phase, size=300, num_classes=21):
    print("WARNINNG::TEMPORARY  VERSION")
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Currently SSD300 models are supported!")
        return
    base = mobilenet_v2
    FEATURE_LAYER = [[13, 17, 'S', 'S', 'S', 'S'], [96, 320, 512, 256, 256, 128]]

    model = build_ssd_lite(phase=phase, size=size, base=base, feature_layer=FEATURE_LAYER, mbox=number_box, num_classes=num_classes)

    return model

def test(device=None):
    if device == "cpu":
        print('CPU mode')
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    net = build_net('train', 300, 21).to(device)
    print(net)
    from torchsummary import summary
    summary(net, input_size=(3, 300, 300), device=str(device))
    inputs = torch.randn(32, 3, 300, 300)
    out = net(inputs.to(device))
    print('coords output size: ', out[0].size())
    print('class output size: ', out[1].size())

#test("cpu")

'''
Total params: 4,577,792
Trainable params: 4,577,792
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.03
Forward/backward pass size (MB): 300.86
Params size (MB): 17.46
Estimated Total Size (MB): 319.35
----------------------------------------------------------------
coords output size:  torch.Size([32, 2990, 4])
class output size:  torch.Size([32, 2990, 21])
'''