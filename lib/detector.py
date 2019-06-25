from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from utils.timer import Timer
from lib.functions import *

class ObjectDetector:
    def __init__(self, net, priorbox, priors, transform, detector, width, height, alt):
        self.model = net
        self.priorbox = priorbox
        self.priors = priors
        self.transform = transform
        self.detector = detector
        self.width = width
        self.height = height
        self.altitude = alt

    def predict(self, img, threshold=0.6):
        # make sure the input channel is 3 
        assert img.shape[2] == 3
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        
        _t = {'inference': Timer(), 'misc': Timer(), 'total': Timer()}
        
        # preprocess image
        _t['total'].tic()
        with torch.no_grad():
            x = self.transform(img).unsqueeze(0).cuda() # for fastening
            
            #x = self.transform(img).unsqueeze(0)
            #if args.cuda:
            #    x = x.cuda()
            #    scale = scale.cuda()

        # forward
        _t['inference'].tic()
        out = self.model(x)  # forward pass
        inference_time = _t['inference'].toc()

        # detect
        _t['misc'].tic()
        detections = self.detector.forward(out, self.priors)
        
        # output
        labels, scores, coords = [list() for _ in range(3)]
        batch=0
        for classes in range(detections.size(1)):
            num = 0
            while detections[batch,classes,num,0] >= threshold:
                scores.append(detections[batch,classes,num,0])
                labels.append(classes-1)
                coords.append(detections[batch,classes,num,1:]*scale)
                num+=1
        misc_time = _t['misc'].toc()
        total_time = _t['total'].toc()

        return labels, scores, coords, (total_time, inference_time, misc_time)