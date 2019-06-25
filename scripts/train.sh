#!/bin/bash

{
    echo 'Training on DRFB mobilenet v1 2'
    python train.py -v DRFB_mobile_2 -d custom --basenet=./weights/mobilenet_feature.pth -max 150
}
{
    echo 'Training on DRFB mobilenet v1'
    python train.py -v DRFB_mobile -d custom --basenet=./weights/mobilenet_feature.pth -max 150
}