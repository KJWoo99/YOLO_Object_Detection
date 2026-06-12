"""Training - YOLO"""
import os
from ultralytics import YOLO
import ultralytics

DATA_ROOT = './data/car_damage'
MODEL     = 'yolo11n-seg.pt'
EPOCHS    = 100
BATCH     = 16
YAML_PATH = os.path.join(DATA_ROOT, 'car_damage.yaml')

if __name__ == '__main__':
    ultralytics.checks()
    model = YOLO(MODEL)
    model.train(
        data    = YAML_PATH,
        epochs  = EPOCHS,
        batch   = BATCH,
        imgsz   = 1280,
        device  = 0,
        patience= 20,
        name    = 'car_damage',
    )
