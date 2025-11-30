# Student Engagement System

**Đồ án xử lý ảnh số - Nhóm 16**

Hệ thống phân loại mức độ hứng thú học tập của sinh viên trong lớp học bằng phân loại khuôn mặt.

## 📋 Mô tả dự án

Dự án sử dụng các kỹ thuật xử lý ảnh và học máy để:
1. **Phát hiện khuôn mặt** trong ảnh/video lớp học
2. **Nhận dạng cảm xúc** từ khuôn mặt đã phát hiện
3. **Phân loại mức độ hứng thú** học tập dựa trên cảm xúc

### Các mức độ hứng thú

| Mức độ | Tiếng Anh | Mô tả |
|--------|-----------|-------|
| 🟢 Rất hứng thú | Highly Engaged | Sinh viên tập trung cao, biểu hiện tích cực |
| 🔵 Hứng thú | Engaged | Sinh viên quan tâm, chú ý |
| 🟡 Bình thường | Neutral | Trạng thái bình thường |
| 🟠 Không hứng thú | Disengaged | Sinh viên mất tập trung |
| 🔴 Rất không hứng thú | Highly Disengaged | Sinh viên hoàn toàn không quan tâm |

## 📁 Cấu trúc thư mục

```
Student_Engagement_System/
│
├── configs/                    # Cấu hình hệ thống
│   └── config.yaml             # File cấu hình chính
│
├── data/                       # Dữ liệu
│   ├── raw/                    # Dữ liệu thô (ảnh/video gốc)
│   ├── processed/              # Dữ liệu đã xử lý
│   └── models/                 # Models đã train
│
├── notebooks/                  # Jupyter notebooks
│   └── README.md               # Hướng dẫn notebooks
│
├── src/                        # Source code chính
│   ├── __init__.py
│   ├── face_detection/         # Module phát hiện khuôn mặt
│   │   ├── __init__.py
│   │   └── detector.py         # Các thuật toán phát hiện
│   │
│   ├── emotion_recognition/    # Module nhận dạng cảm xúc
│   │   ├── __init__.py
│   │   └── recognizer.py       # Các thuật toán nhận dạng
│   │
│   ├── engagement_classifier/  # Module phân loại hứng thú
│   │   ├── __init__.py
│   │   └── classifier.py       # Logic phân loại
│   │
│   ├── data_processing/        # Module xử lý dữ liệu
│   │   ├── __init__.py
│   │   └── data_loader.py      # Tải và tiền xử lý dữ liệu
│   │
│   └── visualization/          # Module hiển thị kết quả
│       ├── __init__.py
│       └── visualizer.py       # Vẽ kết quả lên ảnh/video
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_face_detection.py
│   └── test_engagement_classifier.py
│
├── utils/                      # Các hàm tiện ích
│   ├── __init__.py
│   └── helpers.py
│
├── main.py                     # Entry point chính
├── requirements.txt            # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Webcam (cho chế độ real-time)

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/23133027-HuuHuy/Student_Engagement_System.git
cd Student_Engagement_System

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 💻 Sử dụng

### Chế độ Camera (Real-time)

```bash
python main.py --mode camera --camera 0
```

### Chế độ Ảnh

```bash
python main.py --mode image --input path/to/image.jpg --output result.jpg
```

### Tùy chọn

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--mode` | Chế độ: `camera` hoặc `image` | `camera` |
| `--config` | Đường dẫn file cấu hình | `configs/config.yaml` |
| `--input` | Ảnh đầu vào (cho mode image) | - |
| `--output` | Ảnh kết quả (cho mode image) | - |
| `--camera` | ID camera | `0` |

## 🧪 Chạy tests

```bash
pytest tests/ -v
```

## 📚 Các module chính

### 1. Face Detection (`src/face_detection/`)
- Phát hiện khuôn mặt sử dụng Haar Cascade, dlib, hoặc MediaPipe
- Trả về bounding boxes của các khuôn mặt

### 2. Emotion Recognition (`src/emotion_recognition/`)
- Nhận dạng 7 cảm xúc cơ bản: happy, sad, angry, surprise, fear, disgust, neutral
- Sử dụng CNN hoặc pre-trained models

### 3. Engagement Classifier (`src/engagement_classifier/`)
- Map cảm xúc sang mức độ hứng thú
- Tính toán thống kê cho cả lớp

### 4. Visualization (`src/visualization/`)
- Vẽ bounding boxes và labels lên ảnh
- Hiển thị biểu đồ cảm xúc

## 🔧 Cấu hình

Chỉnh sửa file `configs/config.yaml` để tùy chỉnh:
- Phương pháp phát hiện khuôn mặt
- Mapping cảm xúc-hứng thú
- Cài đặt camera
- Tham số huấn luyện

## 👥 Thành viên nhóm

- Nhóm 16 - Đồ án Xử lý ảnh số

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.
