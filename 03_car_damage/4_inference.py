"""Inference visualization - instance segmentation mask overlay"""
import os, glob, random, cv2, numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from ultralytics import YOLO

CLASS_NAMES = ['Scratched', 'Breakage', 'Separated', 'Crushed']
DATA_ROOT   = './data/car_damage'
BEST_PT     = os.path.join('runs', 'segment', 'car_damage', 'weights', 'best.pt')

model     = YOLO(BEST_PT)
test_imgs = glob.glob(os.path.join(DATA_ROOT, 'split', 'test', 'images', '*'))
random.shuffle(test_imgs)
colors    = [tuple(random.randint(50,200) for _ in range(3)) for _ in CLASS_NAMES]

plt.figure(figsize=(18, 12))
for idx in range(min(6, len(test_imgs))):
    img_bgr = cv2.imread(test_imgs[idx])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = model(img_bgr)[0]
    overlay = img_rgb.copy()
    if result.masks is not None:
        for i, m in enumerate(result.masks):
            cls  = int(result.boxes.cls[i].item())
            poly = m.xy[0].astype(np.int32)
            mask = polygon2mask(img_bgr.shape[:2], poly[:, ::-1])
            overlay[mask.astype(bool)] = colors[cls]
            cv2.polylines(overlay, [poly], True, colors[cls], 2)
            if len(poly): cv2.putText(overlay, CLASS_NAMES[cls], tuple(poly[0]),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls], 2)
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    plt.subplot(2, 3, idx+1); plt.imshow(blended); plt.axis('off')
plt.suptitle('Car Damage Segmentation - Inference')
os.makedirs('result', exist_ok=True)
plt.tight_layout(); plt.savefig('result/inference_result.png', dpi=120); plt.close()
