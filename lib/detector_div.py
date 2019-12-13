from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from utils.timer import Timer
from lib.functions import *

class ObjectDetector_div:
    def __init__(self, net, priorbox, priors, transform, detector, width, height, alt):
        self.model = net
        self.priorbox = priorbox
        self.priors = priors
        self.transform = transform
        self.detector = detector
        self.width = width
        self.height = height
        self.half = width // 2
        self.over_area = int(width*0.0625)
        self.scale_half = torch.Tensor([self.half+self.over_area, height, self.half+self.over_area, height])
        self.middle_coords = [self.half-self.over_area, 0, self.half+self.over_area, height]
        self.altitude = int(alt)
        self.th_x, self.th_y = th_selector(self.over_area, self.altitude)

    def predict(self, img, threshold=0.6):
        # make sure the input channel is 3 
        assert img.shape[2] == 3

        _t = {'inference': Timer(), 'misc': Timer(), 'total': Timer()}

        l_middle_objs = []
        
        img_l = img[:, :self.half+self.over_area]
        img_r = img[:, self.half-self.over_area:]
        
        # preprocess image
        _t['total'].tic()
        with torch.no_grad():
            x_l = self.transform(img_l).unsqueeze(0).cuda() # for fastening
            x_r = self.transform(img_r).unsqueeze(0).cuda() # for fastening
            
            #x_l = self.transform(img_l).unsqueeze(0)
            #x_r = self.transform(img_r).unsqueeze(0)
            #if args.cuda:
            #    x = x.cuda()
            #    scale = scale.cuda()

        # forward
        _t['inference'].tic()
        out_l = self.model(x_l)  # forward pass
        out_r = self.model(x_r)
        inference_time = _t['inference'].toc()

        # detect
        _t['misc'].tic()
        detections_l = self.detector.forward(out_l, self.priors)
        detections_r = self.detector.forward(out_r, self.priors)
        
        labels, scores, coords = [list() for _ in range(3)]
        
        # left objects
        batch=0
        for classes in range(detections_l.size(1)):
            num = 0
            while detections_l[batch,classes,num,0] >= threshold:
                
                t_bbox = detections_l[batch,classes,num,1:]*self.scale_half
                sx = int(t_bbox[0])
                sy = int(t_bbox[1])
                ex = int(t_bbox[2])
                ey = int(t_bbox[3])
                bbox = [sx, sy, ex, ey]

                if is_overlap_area(self.middle_coords, bbox):
                    bbox.append(classes-1)
                    bbox.append(detections_l[batch,classes,num,0])
                    l_middle_objs.append(bbox)
                else:
                    scores.append(detections_l[batch,classes,num,0])
                    labels.append(classes-1)
                    coords.append(detections_l[batch,classes,num,1:]*self.scale_half)
                num+=1
        
        # right objects
        batch=0
        for classes in range(detections_r.size(1)):
            num = 0
            while detections_r[batch,classes,num,0] >= threshold:
                t_bbox = detections_r[batch,classes,num,1:]*self.scale_half
                sx = int(self.half - self.over_area + t_bbox[0])
                sy = int(t_bbox[1])
                ex = int(self.half - self.over_area + t_bbox[2])
                ey = int(t_bbox[3])
                bbox = [sx, sy, ex, ey]
                if is_overlap_area(self.middle_coords, bbox):
                    bbox.append(classes-1)
                    bbox.append(detections_r[batch,classes,num,0])
                    get_close_obj(l_middle_objs, bbox, self.over_area, self.th_x, self.th_y)
                else:
                    scores.append(detections_r[batch,classes,num,0])
                    labels.append(classes-1)
                    coords.append(bbox)
                num+=1
        
        # middle objects
        for bbox in l_middle_objs:
            coords.append(bbox[:4])
            labels.append(bbox[4])
            scores.append(bbox[5])

        misc_time = _t['misc'].toc()
        total_time = _t['total'].toc()
        
        return labels, scores, coords, (total_time, inference_time, misc_time)