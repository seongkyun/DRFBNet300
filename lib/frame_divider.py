import sys
import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Video to frame')

parser.add_argument('-f', '--file', default=None, help='file to run demo')
parser.add_argument('--save_folder', default='demofiles/', type=str,
                    help='Dir to save results')
parser.add_argument('-r', '--rate', default=1, type=int, help='sampling number')
parser.add_argument('-tf', '--total_frame', default=None, type=int, help='number of total frame')
parser.add_argument('-s', '--start_number', default=0, type=int, help='start number')
args = parser.parse_args()

assert os.path.isfile(args.file), 'ERROR::FILE DOES NOT EXIST'
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def frame_divider(video, rate, start_num, tot_frame, save_dir):
     index = -1
     idx_save = 0
     print('Dividing video to frame...')
     while(video.isOpened()):
          index = index + 1

          flag, img = video.read()
          if flag == False:
               break
          if tot_frame != None:
               if idx_save > tot_frame:
                    break
          if index % rate == 0:
               status = 'Frame count: {:d}, Saving count: {:d} \r'.format(index, idx_save)
               cv2.imwrite(os.path.join(save_dir, 'frame_{}.jpg'.format(idx_save + start_num)), img)
               idx_save = idx_save + 1
          else:
               status = 'Frame count: {:d}, Saving count: {:d} \r'.format(index, idx_save)
          sys.stdout.write(status)
          sys.stdout.flush()

     print('')
     print('Number of total frames: ', index + 1)
     print('Number of saved frames: ', idx_save + 1)
     video.release()
     cv2.destroyAllWindows()

if __name__ == '__main__':
     path, _ = os.path.splitext(args.file)
     filename = 'frame_' + 'rate_' + str(args.rate) + '_' + path.split('/')[-1]
     save_dir = os.path.join(args.save_folder, filename)
     
     if not os.path.exists(save_dir):
          os.mkdir(save_dir)

     video = cv2.VideoCapture(args.file)
     frame_divider(video, args.rate, args.start_number, args.total_frame, save_dir)