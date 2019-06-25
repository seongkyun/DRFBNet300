# Real-time small object detection model in the bird-view UAV imagery
By Seongkyun Han, Juwon Kwon, Soonchul Kwon

- This repo is came from [RFBNet](https://github.com/ruinmessi/RFBNet) and modified for my experiment.

### Introduction

## Installation
- Install [PyTorch-1.0.1](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository. This repository is mainly based on [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Chainer-ssd](https://github.com/Hakuyume/chainer-ssd), a huge thank to them.
  * Note: We currently only support PyTorch-0.4.0 and Python 3+.
- Compile the nms and coco tools:
```Shell
./make.sh
```
*Note*: Check you GPU architecture support in utils/build.py, line 131. Default is:
``` 
'nvcc': ['-arch=sm_52',
```
- Then download the dataset by following the [instructions](#download-voc2007-trainval--test) below and install opencv. 
```Shell
conda install opencv
```
Note: For training, we currently  support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://mscoco.org/). 

## Datasets
To make things easy, we provide simple VOC and COCO dataset loader that inherits `torch.utils.data.Dataset` making it fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

### VOC Dataset
##### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```
### COCO Dataset
Install the MS COCO dataset at /path/to/coco from [official website](http://mscoco.org/), default is ~/data/COCO. Following the [instructions](https://github.com/rbgirshick/py-faster-rcnn/blob/77b773655505599b94fd8f3f9928dbf1a9a776c7/data/README.md) to prepare *minival2014* and *valminusminival2014* annotations. All label files (.json) should be under the COCO/annotations/ folder. It should have this basic structure
```Shell
$COCO/
$COCO/cache/
$COCO/annotations/
$COCO/images/
$COCO/images/test2015/
$COCO/images/train2014/
$COCO/images/val2014/
```
*UPDATE*: The current COCO dataset has released new *train2017* and *val2017* sets which are just new splits of the same image sets. 

## Training
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:    https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
or from our [BaiduYun Driver](https://pan.baidu.com/s/1jIP86jW) 
- MobileNet pre-trained basenet is ported from [MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe), which achieves slightly better accuracy rates than the original one reported in the [paper](https://arxiv.org/abs/1704.04861), weight file is available at: https://drive.google.com/open?id=13aZSApybBDjzfGIdqN1INBlPsddxCK14 or [BaiduYun Driver](https://pan.baidu.com/s/1dFKZhdv).

- By default, we assume you have downloaded the file in the `DRFBNet300/weights` dir:
```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train RFBNet using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.
```Shell
python train_RFB.py -d dataset_name -v version_name -s 300 
```
- Note:
  * -d: choose datasets, VOC or COCO, custom.
  * -v: choose backbone version, RFB_VGG, RFB_E_VGG or RFB_mobile.
  * -s: image size, 300 or 512.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train_RFB.py` for options)
  * If you want to reproduce the results in the paper, the VOC model should be trained about 240 epoches while the COCO version need 130 epoches.
  
## Evaluation
To evaluate a trained network:

```Shell
python test.py -d VOC -v RFB_vgg -s 300 --trained_model /path/to/model/weights
```
By default, it will directly output the mAP results on VOC2007 *test* or COCO *minival2014*. For VOC2012 *test* and COCO *test-dev* results, you can manually change the datasets in the `test_RFB.py` file, then save the detection results and submitted to the server. 


