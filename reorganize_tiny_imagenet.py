import os
import sys
import shutil
import argparse

# Get Tiny ImageNet at https://tiny-imagenet.herokuapp.com/
# This script will help to reorder validation set by folders, so PyTorch's ImageFolder will understand the structure.

IMAGES_DIR = 'images'
ANNOTATION = 'val_annotations.txt'

parser = argparse.ArgumentParser(description='Tiny ImageNet has shitty structure. This script will change it a little.')
parser.add_argument('--inp', default='', type=str, help='Folder with val images', dest='inp')
parser.add_argument('--out', default='', type=str, help='Folder with reorganized val images', dest='out')
args = parser.parse_args()

if not os.path.exists(args.inp):
    print('Cannot find input folder at {}'.format(args.inp))
    sys.exit(1)
if os.path.exists(args.out):
    print('Output folder exists at {}'.format(args.out))
    print('Terminate for safety.')
    sys.exit(1)

print('Inspecting input folder...')
inp_images_dir = os.path.join(args.inp, IMAGES_DIR)
inp_images = os.listdir(inp_images_dir)
print('  Found {:d} files.')
if len(inp_images) == 0:
    print('Input folder is empty. Abort.')
    sys.exit(1)

print('Reading annotations...')
annotation_path = os.path.join(args.inp, ANNOTATION)
filename2label = {}
lines = open(annotation_path, 'r').readlines()
for line in lines:
    items = line.split('\t')
    filename = items[0]
    label = items[1]
    filename2label[filename] = label

print('Creating output folder...')
os.mkdir(args.out)

print('Reorganizing...')
for filename in inp_images:
    src = os.path.join(inp_images_dir, filename)
    label = filename2label[filename]
    dst_dir = os.path.join(args.out, label)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst = os.path.join(dst_dir, filename)
    shutil.copyfile(src, dst)
    print('  Copied {} -> {}'.format(src, dst))

print('Done.')


