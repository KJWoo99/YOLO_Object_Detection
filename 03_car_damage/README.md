# Car Damage Segmentation

YOLO11n-seg 기반 차량 파손 영역 인스턴스 세그멘테이션 모델입니다.  
JSON 폴리곤 어노테이션을 YOLO 세그멘테이션 포맷으로 변환하는 전처리 파이프라인을 포함합니다.

## Dataset

| 항목 | 내용 |
|------|------|
| 출처 | AI Hub 차량 손상 데이터셋 |
| 클래스 | Scratched (스크래치) / Breakage (파손) / Separated (이격) / Crushed (찌그러짐) |
| 어노테이션 | JSON polygon → YOLO segmentation 변환 |
| 분할 | 8:1:1 (train / val / test) |
| 입력 크기 | 1280×1280 |

## Model

- **YOLO11n-seg** (ultralytics, instance segmentation)
- Epochs: 100 (patience=20, EarlyStopping)
- Batch: 16
- imgsz: 1280

## Pipeline

```
1_prepare_data.py  → JSON 파싱 → YOLO seg txt 변환 → 8:1:1 분할 → dataset.yaml 생성
2_train.py         → YOLO11n-seg 학습
3_evaluate.py      → mAP box + mAP seg 평가
4_inference.py     → polygon mask overlay 시각화
```

## Usage

```bash
pip install ultralytics opencv-python pyyaml

python 1_prepare_data.py
python 2_train.py
python 3_evaluate.py
python 4_inference.py
```
