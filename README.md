# Student Engagement System
# Hệ thống Phân loại Mức độ Hứng thú Học tập của Sinh viên

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Đồ án Xử lý Ảnh số - Nhóm 16**

## 📖 Giới thiệu

**Student Engagement System** là hệ thống phân loại mức độ hứng thú học tập của sinh viên trong lớp học sử dụng công nghệ nhận diện khuôn mặt và phân loại cảm xúc.

### Tính năng chính

- 🎯 **Phát hiện khuôn mặt**: Sử dụng Haar Cascade, DNN hoặc MTCNN
- 😊 **Phân loại cảm xúc**: Nhận diện 8 trạng thái cảm xúc
- 📊 **Đánh giá mức độ hứng thú**: 5 mức độ từ "Rất hứng thú" đến "Rất không hứng thú"
- 📹 **Hỗ trợ real-time**: Xử lý video từ webcam hoặc file video
- 📈 **Thống kê và báo cáo**: Theo dõi xu hướng theo thời gian

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- OpenCV 4.5+
- Webcam (cho chế độ real-time)

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/your-username/Student_Engagement_System.git
cd Student_Engagement_System

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📁 Cấu trúc dự án

```
Student_Engagement_System/
├── src/                          # Source code chính
│   ├── core/                     # Các module lõi
│   │   ├── __init__.py
│   │   ├── face_detector.py      # Phát hiện khuôn mặt
│   │   ├── emotion_classifier.py # Phân loại cảm xúc
│   │   └── engagement_analyzer.py # Phân tích mức độ hứng thú
│   │
│   ├── models/                   # Các mô hình ML/DL
│   │   ├── __init__.py
│   │   ├── cnn_model.py          # CNN cho emotion classification
│   │   └── engagement_model.py    # Model dự đoán engagement
│   │
│   ├── data/                     # Xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Tải dữ liệu
│   │   ├── preprocessor.py       # Tiền xử lý
│   │   └── dataset.py            # Dataset class
│   │
│   ├── utils/                    # Tiện ích
│   │   ├── __init__.py
│   │   ├── image_utils.py        # Xử lý ảnh
│   │   ├── video_utils.py        # Xử lý video
│   │   ├── visualization.py      # Hiển thị
│   │   └── logger.py             # Logging
│   │
│   └── config/                   # Cấu hình
│       ├── __init__.py
│       └── settings.py           # Cài đặt hệ thống
│
├── data/                         # Thư mục dữ liệu
│   ├── raw/                      # Dữ liệu thô
│   ├── processed/                # Dữ liệu đã xử lý
│   └── models/                   # Mô hình đã train
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
│
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Tài liệu
├── static/                       # Static files (CSS, JS)
├── templates/                    # HTML templates
│
├── main.py                       # Entry point chính
├── train.py                      # Script training
├── requirements.txt              # Dependencies
└── README.md                     # Tài liệu hướng dẫn
```

## 🎮 Sử dụng

### Chạy với webcam (Real-time)

```bash
python main.py --mode camera
```

### Xử lý file video

```bash
python main.py --mode video --source path/to/video.mp4
```

### Xử lý ảnh đơn

```bash
python main.py --mode image --source path/to/image.jpg --output result.jpg
```

### Các tùy chọn

| Tùy chọn | Mô tả | Mặc định |
|----------|-------|----------|
| `--mode` | Chế độ: camera, video, image | camera |
| `--source` | Nguồn video/ảnh | 0 (webcam) |
| `--output` | Đường dẫn output | None |
| `--debug` | Bật chế độ debug | False |
| `--no-display` | Không hiển thị | False |

## 🎓 Mức độ hứng thú

Hệ thống phân loại 5 mức độ hứng thú:

| Mức độ | Điểm | Màu |
|--------|------|-----|
| Rất hứng thú | 80-100 | 🟢 Xanh lá |
| Hứng thú | 60-79 | 🟢 Xanh nhạt |
| Bình thường | 40-59 | 🟡 Vàng |
| Không hứng thú | 20-39 | 🟠 Cam |
| Rất không hứng thú | 0-19 | 🔴 Đỏ |

## 😊 Các cảm xúc được nhận diện

- Happy (Vui vẻ)
- Sad (Buồn)
- Angry (Tức giận)
- Surprised (Ngạc nhiên)
- Neutral (Trung tính)
- Fear (Sợ hãi)
- Disgust (Ghê tởm)
- Confused (Bối rối)

## 🔧 Training mô hình

### Chuẩn bị dữ liệu

Tổ chức dữ liệu theo cấu trúc:
```
data/raw/
├── happy/
│   ├── img001.jpg
│   └── ...
├── sad/
│   └── ...
└── ...
```

### Chạy training

```bash
python train.py --model emotion --data-dir data/raw --epochs 50 --augment
```

## 🧪 Testing

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy test cụ thể
pytest tests/test_core.py -v

# Với coverage
pytest tests/ --cov=src
```

## 📊 API Reference

### FaceDetector

```python
from src.core import FaceDetector

detector = FaceDetector(method="haar")
faces = detector.detect_faces(image)
```

### EmotionClassifier

```python
from src.core import EmotionClassifier

classifier = EmotionClassifier()
result = classifier.classify(face_image)
print(result['emotion'], result['confidence'])
```

### EngagementAnalyzer

```python
from src.core import EngagementAnalyzer

analyzer = EngagementAnalyzer()
metrics = analyzer.analyze(emotion_result)
print(metrics.engagement_level, metrics.engagement_score)
```

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

Dự án được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 👥 Tác giả

**Nhóm 16** - Đồ án Xử lý Ảnh số

## 🙏 Lời cảm ơn

- OpenCV team
- TensorFlow/PyTorch communities
- Các nguồn dataset công khai về cảm xúc khuôn mặt
