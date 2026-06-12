"""Evaluation - mAP metrics"""
import os
from ultralytics import YOLO

BEST_PT = os.path.join('runs', 'detect', 'safety_helmet', 'weights', 'best.pt')

if __name__ == '__main__':
    model   = YOLO(BEST_PT)
    metrics = model.val(split='test', workers=0)
    print(f'mAP50    (box) : {metrics.box.map50:.4f}')
    print(f'mAP50-95 (box) : {metrics.box.map:.4f}')
