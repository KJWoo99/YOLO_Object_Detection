"""Data Preparation - annotation conversion + train/val/test split + YAML"""
import os, glob, random, shutil, yaml
import json
from tqdm import tqdm

CLASS_NAMES = ['Scratched', 'Breakage', 'Separated', 'Crushed']
DATA_ROOT   = './data/car_damage'
SPLIT       = (0.8, 0.1, 0.1)
SEED        = 2024

def convert_annotations(data_root, class_names):
    ann_dir = os.path.join(data_root, 'annotations')
    lbl_dir = os.path.join(data_root, 'labels')
    os.makedirs(lbl_dir, exist_ok=True)
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.json')), desc='Converting'):
        with open(ann_file, 'r') as f: data = json.load(f)
        h = data['images']['height']; w = data['images']['width']
        result = []
        for ann in data.get('annotations', []):
            dmg = ann.get('damage', '')
            if dmg not in class_names: continue
            idx  = class_names.index(dmg)
            poly = ann['segmentation'][0][0][:-1]
            coords = ' '.join(f'{pt[0]/w:.6f} {pt[1]/h:.6f}' for pt in poly)
            result.append(f'{idx} {coords}')
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
    yaml_path = os.path.join(DATA_ROOT, 'car_damage.yaml')
    abs_root  = os.path.abspath(DATA_ROOT).replace('\\', '/')
    with open(yaml_path, 'w') as f:
        yaml.dump({'path' : abs_root,
                   'train': 'split/train',
                   'val'  : 'split/val',
                   'test' : 'split/test',
                   'nc'   : len(CLASS_NAMES), 'names': CLASS_NAMES}, f)
    print(f'YAML saved: {yaml_path}')
