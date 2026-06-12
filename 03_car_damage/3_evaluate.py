"""Evaluation - mAP metrics"""
import os
from ultralytics import YOLO

BEST_PT = os.path.join('runs', 'segment', 'car_damage', 'weights', 'best.pt')

if __name__ == '__main__':
    model   = YOLO(BEST_PT)
    metrics = model.val(split='test', workers=0)
    print(f'mAP50    (box) : {metrics.box.map50:.4f}')
    print(f'mAP50-95 (box) : {metrics.box.map:.4f}')
    print(f'mAP50    (seg) : {metrics.seg.map50:.4f}')
    print(f'mAP50-95 (seg) : {metrics.seg.map:.4f}')
