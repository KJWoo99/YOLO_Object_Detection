"""Data Preparation - annotation conversion + train/val/test split + YAML"""
import os, glob, random, shutil, yaml
import xml.etree.ElementTree as ET
from tqdm import tqdm

CLASS_NAMES = ['D00', 'D10', 'D20', 'D40', 'D43', 'D44', 'D50']
DATA_ROOT   = './data/road_crack'
SPLIT       = (0.8, 0.1, 0.1)
SEED        = 2024

def convert_annotations(data_root, class_names):
    ann_dir = os.path.join(data_root, 'annotations')
    lbl_dir = os.path.join(data_root, 'labels')
    os.makedirs(lbl_dir, exist_ok=True)
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '**', '*.xml'), recursive=True), desc='Converting'):
        tree = ET.parse(ann_file)
        root = tree.getroot()
        w = int(root.find('size').find('width').text)
        h = int(root.find('size').find('height').text)
        result = []
        for obj in root.findall('object'):
            lbl = obj.find('name').text
            if lbl not in class_names: continue
            idx  = class_names.index(lbl)
            bbox = [int(x.text) for x in obj.find('bndbox')]
            xc   = ((bbox[2]+bbox[0])/2)/w
            yc   = ((bbox[3]+bbox[1])/2)/h
            bw   = (bbox[2]-bbox[0])/w
            bh   = (bbox[3]-bbox[1])/h
            result.append(f'{idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}')
        if result:
            with open(os.path.join(lbl_dir, os.path.splitext(os.path.basename(ann_file))[0]+'.txt'), 'w') as f:
                f.write('\n'.join(result))

def split_dataset(data_root):
    img_files = glob.glob(os.path.join(data_root, 'images', '*'))
    random.seed(SEED); random.shuffle(img_files)
    n = len(img_files)
    cuts = {
        'test' : img_files[:int(n*SPLIT[2])],
        'val'  : img_files[int(n*SPLIT[2]):int(n*(SPLIT[1]+SPLIT[2]))],
        'train': img_files[int(n*(SPLIT[1]+SPLIT[2])):],
    }
    for split, files in cuts.items():
        for sub in ['images', 'labels']: os.makedirs(os.path.join(data_root, 'split', split, sub), exist_ok=True)
        for img_path in files:
            fname = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(data_root, 'split', split, 'images', fname))
            lbl = os.path.join(data_root, 'labels', os.path.splitext(fname)[0]+'.txt')
            if os.path.exists(lbl):
                shutil.copy(lbl, os.path.join(data_root, 'split', split, 'labels', os.path.splitext(fname)[0]+'.txt'))
    print({k: len(v) for k, v in cuts.items()})

if __name__ == '__main__':
    convert_annotations(DATA_ROOT, CLASS_NAMES)
    split_dataset(DATA_ROOT)
    yaml_path = os.path.join(DATA_ROOT, 'road_crack.yaml')
    abs_root  = os.path.abspath(DATA_ROOT).replace('\\', '/')
    with open(yaml_path, 'w') as f:
        yaml.dump({'path' : abs_root,
                   'train': 'split/train',
                   'val'  : 'split/val',
                   'test' : 'split/test',
                   'nc'   : len(CLASS_NAMES), 'names': CLASS_NAMES}, f)
    print(f'YAML saved: {yaml_path}')
