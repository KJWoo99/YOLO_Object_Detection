"""Training - YOLO"""
import os
from ultralytics import YOLO
import ultralytics

DATA_ROOT = './data/road_crack'
MODEL     = 'yolov8n.pt'
EPOCHS    = 100
BATCH     = 16
YAML_PATH = os.path.join(DATA_ROOT, 'road_crack.yaml')

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
        rect    = True,
        name    = 'road_crack',
    )
