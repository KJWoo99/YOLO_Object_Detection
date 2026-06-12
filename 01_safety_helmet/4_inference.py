"""Inference visualization - bounding box detection"""
import os, glob, random, cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

CLASS_NAMES = ['helmet', 'head', 'person']
DATA_ROOT   = './data/safety_helmet'
BEST_PT     = os.path.join('runs', 'detect', 'safety_helmet', 'weights', 'best.pt')

model      = YOLO(BEST_PT)
test_imgs  = glob.glob(os.path.join(DATA_ROOT, 'split', 'test', 'images', '*'))
random.shuffle(test_imgs)
colors     = [tuple(random.randint(0,255) for _ in range(3)) for _ in CLASS_NAMES]

plt.figure(figsize=(18, 12))
for idx in range(min(6, len(test_imgs))):
    img_bgr = cv2.imread(test_imgs[idx])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = model(img_bgr)[0]
    ann     = Annotator(img_rgb)
    for box in result.boxes:
        ann.box_label(box.xyxy[0], CLASS_NAMES[int(box.cls)], colors[int(box.cls)])
    plt.subplot(2, 3, idx+1); plt.imshow(ann.result()); plt.axis('off')
plt.suptitle('Safety Helmet Detection - Inference')
os.makedirs('result', exist_ok=True)
plt.tight_layout(); plt.savefig('result/inference_result.png', dpi=120); plt.close()
