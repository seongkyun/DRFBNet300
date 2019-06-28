import os
import sys
import shutil
import argparse

parser = argparse.ArgumentParser(description='Merge images into one directory')

parser.add_argument('-f', '--folder_dir', default=None, help='folders')
parser.add_argument('--dest_dir', default=None, type=str,
                    help='Dir to save images')
args = parser.parse_args()

if not os.path.exists(args.dest_dir):
    print("ERROR::DEST_DIR DOES NOT EXIST")
    sys.exit()

folder_list = []
for i in os.listdir(args.folder_dir):
    folder_list.append(i)

folder_list = sorted(folder_list)

src_path = None
img_path = None
dest_path = None
filename = None

index = -1

print('In processing...')

for i in folder_list:
    print('Current folder: ', i)
    src_path = os.path.join(args.folder_dir, i)
    images = sorted(os.listdir(src_path))
    for j in images:
        index += 1
        filename = "%05d.jpg"%index
        img_path = os.path.join(src_path, j)
        dest_path = os.path.join(args.dest_dir, filename)
        shutil.copy(img_path, dest_path)
print('Process is done.')





    

    
    

