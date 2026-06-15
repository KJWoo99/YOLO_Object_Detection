# Safety Helmet Detection

YOLOv8n 기반 작업 현장 안전모 착용 여부 실시간 탐지 모델입니다.

## Dataset

| 항목 | 내용 |
|------|------|
| 클래스 | helmet (착용) / head (미착용) / person |
| 어노테이션 | Pascal VOC XML → YOLO bbox 변환 |
| 분할 | 8:1:1 (train / val / test) |
| 입력 크기 | 1280×1280 |

## Model

- **YOLOv8n** (ultralytics)
- Epochs: 100 (patience=20, EarlyStopping)
- Batch: 16
- imgsz: 1280

## Results

| Metric | Value |
|--------|-------|
| Precision | 0.631 |
| Recall | 0.609 |
| mAP50 | 0.633 |
| mAP50-95 | 0.418 |

> YOLOv8n (nano) 기준, Epoch 89 (EarlyStopping) 조기 종료 결과.

## Pipeline

```
1_prepare_data.py  → XML 파싱 → YOLO txt 변환 → 8:1:1 분할 → dataset.yaml 생성
2_train.py         → YOLOv8n 학습
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
