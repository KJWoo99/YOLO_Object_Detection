# YOLO Object Detection

YOLOv8n / YOLO11n 기반 객체 탐지 및 인스턴스 세그멘테이션 프로젝트 3종.  
Pascal VOC XML·JSON 어노테이션을 YOLO 포맷으로 변환하는 전처리 파이프라인부터 학습·평가·추론까지 구현.

---

## 프로젝트

### 1. 안전모 착용 감지 (Safety Helmet Detection)

작업 현장 이미지에서 안전모 착용 여부를 탐지.

- **모델**: YOLOv8n (detection)
- **클래스**: helmet / head (미착용) / person
- **어노테이션**: Pascal VOC XML → YOLO bbox
- **설정**: imgsz 1280, epochs 100, patience 20
- **결과**: mAP50 **0.633** / mAP50-95 **0.418**

### 2. 도로 파손 감지 (Road Crack Detection)

도로 이미지에서 균열·포트홀 등 7종 노면 손상을 탐지.

- **모델**: YOLOv8n (detection)
- **클래스**: D00 (종방향 균열) / D10 (횡방향 균열) / D20 (망상 균열) / D40 (포트홀) / D43 / D44 / D50
- **어노테이션**: Pascal VOC XML → YOLO bbox
- **설정**: imgsz 1280, epochs 100, patience 20, rect=True
- **데이터**: RoadDamageDetector (sekilab)
- **결과**: mAP50 **0.612** / mAP50-95 **0.300**

### 3. 차량 파손 세그멘테이션 (Car Damage Segmentation)

차량 이미지에서 파손 영역을 폴리곤 마스크로 정밀 세그멘테이션.

- **모델**: YOLO11n-seg (instance segmentation)
- **클래스**: Scratched (스크래치) / Breakage (파손) / Separated (이격) / Crushed (찌그러짐)
- **어노테이션**: JSON polygon → YOLO segmentation
- **설정**: imgsz 640, epochs 100, patience 20
- **데이터**: AI Hub 차량 손상 데이터셋

---

## 공통 파이프라인

```
1. 어노테이션 변환   Pascal VOC XML / JSON → YOLO format (.txt)
2. 데이터 분할       8:1:1 (train / val / test)
3. YAML 생성         dataset.yaml (paths + class names)
4. 학습              model.train(epochs=100, patience=20)
5. 평가              mAP50 / mAP50-95
6. 추론 시각화       bounding box (1·2) / polygon mask overlay (3)
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Framework | Ultralytics YOLO |
| Models | YOLOv8n (detection), YOLO11n-seg (segmentation) |
| Annotation | Pascal VOC XML, JSON polygon |
| Library | OpenCV, scikit-image, PyYAML |

---

## 디렉토리 구조

```
.
├── 01_safety_helmet/
│   ├── README.md
│   ├── 1_prepare_data.py   # XML → YOLO bbox + 분할 + YAML 생성
│   ├── 2_train.py          # YOLOv8n 학습
│   ├── 3_evaluate.py       # mAP50 / mAP50-95
│   ├── 4_inference.py      # bbox 시각화
│   └── result/             # inference_result.png
├── 02_road_crack/
│   ├── README.md
│   ├── 1_prepare_data.py   # XML → YOLO bbox + 분할 + YAML 생성
│   ├── 2_train.py          # YOLOv8n 학습 (rect=True)
│   ├── 3_evaluate.py
│   ├── 4_inference.py
│   └── result/
└── 03_car_damage/
    ├── README.md
    ├── 1_prepare_data.py   # JSON polygon → YOLO seg + 분할 + YAML 생성
    ├── 2_train.py          # YOLO11n-seg 학습
    ├── 3_evaluate.py       # mAP box + seg
    └── 4_inference.py      # polygon mask overlay 시각화
```

## 실행 순서

```bash
# 각 프로젝트 폴더 안에서 실행
python 1_prepare_data.py   # 데이터 변환 및 분할 (최초 1회)
python 2_train.py          # 학습
python 3_evaluate.py       # mAP 평가
python 4_inference.py      # 추론 시각화
```
