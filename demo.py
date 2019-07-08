from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import *
import cv2
import torch.utils.data as data
from layers.functions import Detect_test,PriorBox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from utils.nms_wrapper import nms
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-f', '--file', default=None, help='file to run demo')
parser.add_argument('-c', '--camera_num', default=0, type=int, 
                    help='demo camera number(default is 0)')
parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='results/', type=str,
                    help='Dir to save results')
parser.add_argument('-th', '--threshold', default=0.45,
                    type=float, help='Detection confidence threshold value')
parser.add_argument('-t', '--type', dest='type', default='image', type=str,
            help='the type of the demo file, could be "image", "video", "camera"')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--div', default=False, type=bool, 
                    help='Use half divided mode')
parser.add_argument('-alt', '--altitude', default=10, help='drone altitude, unit: meter')
args = parser.parse_args()

# Make result file saving folder
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# Object detector setting
if args.div:
    print("Running on divided mode")
    try:
        from lib.detector_div import ObjectDetector_div as ObjectDetector
    except ImportError:
        print("lib folder is not exist.")
        print("Running on common mode")
        from lib.detector import ObjectDetector   
        args.div = False
else:
    print("Running on common mode")
    from lib.detector import ObjectDetector

# Label settings
if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
    from data.voc0712 import VOC_CLASSES
    lable_map = VOC_CLASSES[1:]
    num_classes = 21
elif args.dataset == 'COCO':
    cfg = (COCO_300, COCO_512)[args.size == '512']
    from data.coco import COCO_CLASSES
    lable_map = COCO_CLASSES
    num_classes = 81
else:
    from data.custom_voc import CLASSES
    lable_map = CLASSES[1:]
    num_classes = 2

# Version checking
if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = mobile_300
elif args.version == 'DRFB_mobile':
    from models.DRFB_Net_mobile import build_net
    cfg = mobile_300
elif args.version == 'SSD_vgg':
    from models.SSD_vgg import build_net
    cfg = (VOC_SSDVGG_300, COCO_SSDVGG_300)[args.dataset == 'COCO']
elif args.version == 'SSD_mobile':
    from models.SSD_lite_mobilenet_v1 import build_net
    cfg = mobile_300
elif args.version == 'DRFB_mobile2':
    from models.DRFB_Net_mobile2 import build_net
    cfg = mobile_300
else:
    print('ERROR::UNKNOWN VERSION')
    sys.exit()

# color number book: http://www.n2n.pe.kr/lev-1/color.htm
COLORS = [(0, 0, 204), (153, 255, 51), (255, 204, 0)] # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Prior box setting
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

def demo_img(object_detector, img, save_dir):

    _labels, _scores, _coords, times= object_detector.predict(img, args.threshold)

    FPS = float(1/times[0])
    for labels, scores, coords in zip(_labels, _scores, _coords):
        cv2.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
        cv2.putText(img, '{label}: {score:.2f}'.format(label=lable_map[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 1, COLORS[labels % 3], 1)

    status = 'FPS: {:.2f} T_inf: {:.3f} T_misc: {:.3f}s \r'.format(FPS, times[1], times[2])
    cv2.putText(img, status[:-2], (10, 20), FONT, 0.5, (0, 0, 0), 5)
    cv2.putText(img, status[:-2], (10, 20), FONT, 0.5, (255, 255, 255), 2)
    cv2.imwrite(save_dir, img)

def demo_stream(object_detector, video, save_dir):
    index = -1

    video_dir = os.path.join(save_dir, 'result.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(video_dir, fourcc, 25.0, (object_detector.width,object_detector.height))

    avg_FPS = 0.0
    sum_FPS = 0.0

    while(video.isOpened()):
        index = index + 1
        
        flag, img = video.read()
        if flag == False:
            break

        _labels, _scores, _coords, times= object_detector.predict(img, args.threshold)
        
        sum_FPS += float(1/times[0])
        avg_FPS = sum_FPS / (index+1)

        for labels, scores, coords in zip(_labels, _scores, _coords):
            cv2.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
            cv2.putText(img, '{label}: {score:.2f}'.format(label=lable_map[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 1, COLORS[labels % 3], 2)
    
        status = 'Frame: {:d} FPS: {:.2f} T_inf: {:.3f} T_misc: {:.3f}s \r'.format(index, avg_FPS, times[1], times[2])
        cv2.putText(img, status[:-2], (10, 20), FONT, 0.5, (0, 0, 0), 5)
        cv2.putText(img, status[:-2], (10, 20), FONT, 0.5, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(save_dir, 'frame_{}.jpg'.format(index)), img)
        video_out.write(img)
        sys.stdout.write(status)
        sys.stdout.flush()
    
    video.release()
    video_out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Validity check
    print('Validity check...')
    if not args.type in ['camera', 'folder']:
        assert os.path.isfile(args.file), 'ERROR::DEMO FILE DOES NOT EXIST'
    assert os.path.isfile(args.trained_model), 'ERROR::WEIGHT FILE DOES NOT EXIST'
    file_type = args.file[-3:].lower()
    if file_type in ['png', 'jpg']:
        print('Demo type is changed to the input file type: image')
        args.type = 'image'
    elif file_type in ['mp4', 'avi']:
        print('Demo type is changed to the input file type: video')
        args.type = 'video'

    # Directory setting
    print('Directory setting...')
    if args.type == 'image':
        path, _ = os.path.splitext(args.file)
        filename = args.version + '_' + path.split('/')[-1]
        save_dir = os.path.join(args.save_folder, filename + '.jpg')
        if args.div:
            save_dir = os.path.join(args.save_folder, filename + '_divided_mode.jpg')
    elif args.type == 'video':
        path, _ = os.path.splitext(args.file)
        filename = args.version + '_' + path.split('/')[-1]
        save_dir = os.path.join(args.save_folder, filename)
        if args.div:
            save_dir = save_dir + '_divided_mode'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    elif args.type == 'camera':
        filename = args.version + '_camera_' + str(args.camera_num)
        save_dir = os.path.join(args.save_folder, filename)
        if args.div:
            save_dir = save_dir + '_divided_mode'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    elif args.type == 'folder':
        path, _ = os.path.splitext(args.file)
        filename = args.version + '_' + path.split('/')[-2]
        save_dir = os.path.join(args.save_folder, filename)
        if args.div:
            save_dir = save_dir + '_divided_mode'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    else:
        raise AssertionError('ERROR::TYPE IS NOT CORRECT')

    # Setting network
    print('Network setting...')
    img_dim = (300,512)[args.size=='512']
    rgb_means = ((103.94,116.78,123.68), (104, 117, 123))[args.version == 'RFB_vgg' or args.version == 'RFB_E_vgg']
    p = (0.2, 0.6)[args.version == 'RFB_vgg' or args.version == 'RFB_E_vgg']
    
    print('Loading pretrained model')
    net = build_net('test', 300, num_classes)    # initialize detector
    state_dict = torch.load(args.trained_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    print('Finished loading model')
    
    nms_th = 0.45
    max_det = 100

    print('NMS_th: {:.2f}, Max_det: {:d}, Conf_th: {:.2f}'.format(nms_th, max_det, args.threshold))
    detector = Detect_test(num_classes, 0, cfg, nms_th, args.threshold, max_det, priors)
    transform = BaseTransform(net.size, rgb_means, (2, 0, 1))

    # Running demo
    print('Running demo...')
    if args.type == 'image':
        img = cv2.imread(args.file)
        
        width = int(img.shape[1])
        height = int(img.shape[0])
        object_detector = ObjectDetector(net, priorbox, priors, transform, detector, width, height, args.altitude)
        
        demo_img(object_detector, img, save_dir)
    elif args.type == 'video' or args.type == 'camera':
        video = cv2.VideoCapture(args.file)
        
        width = int(video.get(3))
        height = int(video.get(4))
        object_detector = ObjectDetector(net, priorbox, priors, transform, detector, width, height, args.altitude)
        
        demo_stream(object_detector, video, save_dir)
    elif args.type == 'folder':
        from lib.map_functions import file_list
        img_list = sorted(file_list(args.file))
        
        tot_idx = len(img_list)
        index = -1
        save_folder = save_dir

        for img_file in img_list:
            index += 1
            save_dir = os.path.join(save_folder, img_file)

            img = cv2.imread(os.path.join(args.file, img_file))
            width = int(img.shape[1])
            height = int(img.shape[0])
            object_detector = ObjectDetector(net, priorbox, priors, transform, detector, width, height, args.altitude)

            demo_img(object_detector, img, save_dir)
            status = 'Total Images: {:d} Cur Image: {:d} \r'.format(tot_idx, index+1)
            sys.stdout.write(status)
            sys.stdout.flush()
    else:
        raise AssertionError('ERROR::TYPE IS NOT CORRECT')

