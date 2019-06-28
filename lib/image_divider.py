import os
import sys
import shutil
import argparse

parser = argparse.ArgumentParser(description='Merge images into one directory')

parser.add_argument('-i', '--img_dir', default=None, help='image dir')
parser.add_argument('--dest_dir', default=None, type=str,
                    help='Dir to save images')
parser.add_argument('-r', '--rate', default=2, type=int, help='divider') 
args = parser.parse_args()

if not os.path.exists(args.dest_dir):
    print("ERROR::DEST_DIR DOES NOT EXIST")
    sys.exit()

path = None
paths = []
for j in range(args.rate):
    path = os.path.join(args.dest_dir, "images_"+str(j))
    paths.append(path)
    if not os.path.exists(path):
        os.mkdir(path)
img_list = sorted(os.listdir(args.img_dir))

for i in img_list:
    ori_path = os.path.join(args.img_dir, i)
    number = int(i[:-4])
    final_folder = paths[number%args.rate]
    final_path = os.path.join(final_folder, i)
    shutil.copy(ori_path, final_path)
    status = 'Total images: {:d} Current image: {:s} \r'.format(len(img_list), i)
    sys.stdout.write(status)
    sys.stdout.flush()