# modified from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py
# github: https://github.com/ScanNet/ScanNet/tree/master/SensReader/python
# python 2.7 is recommended.


import argparse
import os, sys

import os
import numpy as np
import argparse
import random
from tqdm import tqdm

from SensorData import SensorData

def parse_raw_data(output_path, data_filename):
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  # load the data
  sys.stdout.write('loading %s...' % data_filename)
  sd = SensorData(data_filename)
  sys.stdout.write('loaded!\n')
  if opt.export_depth_images:
    sd.export_depth_images(os.path.join(output_path, 'depth'))
  if opt.export_color_images:
    sd.export_color_images(os.path.join(output_path, 'color'))
  if opt.export_poses:
    sd.export_poses(os.path.join(output_path, 'pose'))
  if opt.export_intrinsics:
    sd.export_intrinsics(os.path.join(output_path, 'intrinsic'))


# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.set_defaults(export_depth_images=True, export_color_images=True, export_poses=True, export_intrinsics=True)


opt = parser.parse_args()
print(opt)


data_dir = "PATH_TO_SCANNET/ScanNet/scans_val/" # path to list of scannet scenes
val_seqs = os.listdir(data_dir)
with open("PATH_TO_SCANNET/ScanNet/tasks/scannetv2_val.txt") as f:
    val_seq_ids = f.readlines()
    val_seq_ids = [s.strip() for s in val_seq_ids]

for i in tqdm(range(len(val_seqs))):
  val_id = val_seqs[i]
  val_seq_dir = os.path.join(data_dir, val_id, "renders")
  raw_data_filename = os.path.join(data_dir, val_id, val_id+".sens")
  parse_raw_data(val_seq_dir, raw_data_filename)

if __name__ == '__main__':
    main()