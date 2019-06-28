import os
import sys
import argparse
from map_functions import read_content

parser = argparse.ArgumentParser(description='Count objects in the custom dataset')

parser.add_argument('-a', '--anno_dir', default=None, help='annotation dir')
args = parser.parse_args()

anno_list = sorted(os.listdir(args.anno_dir))
paths = []

for j in anno_list:
    paths.append(os.path.join(args.anno_dir, j))

total_imgs = len(paths)
total_objs = 0

for j in paths:
    img_name, boxes, labels = read_content(j)
    total_objs += len(labels)

print("Total images: ", total_imgs, "Total objects: ", total_objs)
