# Road Crack Detection

YOLOv8n 기반 도로 노면 파손 7종 탐지 모델입니다.

## Dataset

| 항목 | 내용 |
|------|------|
| 출처 | RoadDamageDetector (sekilab) |
| 클래스 | D00 (종방향 균열) / D10 (횡방향 균열) / D20 (망상 균열) / D40 (포트홀) / D43 / D44 / D50 |
| 어노테이션 | Pascal VOC XML → YOLO bbox 변환 |
| 분할 | 8:1:1 (train / val / test) |
| 입력 크기 | 1280×1280 |

## Model

- **YOLOv8n** (ultralytics)
- Epochs: 100 (patience=20, EarlyStopping)
- Batch: 16
- imgsz: 1280
- rect: True (직사각형 배치로 패딩 최소화)

## Pipeline

```
1_prepare_data.py  → XML 파싱 → YOLO txt 변환 → 8:1:1 분할 → dataset.yaml 생성
2_train.py         → YOLOv8n 학습 (rect=True)
3_evaluate.py      → mAP50 / mAP50-95 평가
4_inference.py     → bbox 시각화
```

## Usage

```bash
pip install ultralytics opencv-python pyyaml

python 1_prepare_data.py
python 2_train.py
python 3_evaluate.py
python 4_inference.py
```
