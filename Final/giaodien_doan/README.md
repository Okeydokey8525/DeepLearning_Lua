# 🌾 RiceGuard AI – Hệ thống chẩn đoán bệnh lá lúa

**Stack:** Spring Boot 3 (Java 17) · FastAPI (Python 3.10+) · MySQL 8 · YOLOv12-BiFormer-P2

---

## 📁 Cấu trúc dự án

```
giaodien_doan/
├── ai-server/                  ← FastAPI inference server
│   ├── main.py
│   └── requirements.txt
├── src/
│   ├── main/
│   │   ├── java/com/example/giaodien_doan/
│   │   │   ├── config/         ← WebClient, CORS config
│   │   │   ├── controller/     ← REST Endpoints
│   │   │   ├── dto/response/   ← DTO objects
│   │   │   ├── entity/         ← DiagnosisRecord @Entity
│   │   │   ├── repository/     ← Spring Data JPA
│   │   │   └── service/impl/   ← Business logic + WebClient
│   │   └── resources/
│   │       ├── static/
│   │       │   └── index.html  ← Giao diện web (Glassmorphism)
│   │       └── application.properties
├── pom.xml
└── README.md                   ← File hướng dẫn chạy dự án này
```

---

## 🚀 Hướng dẫn chạy

### Bước 1 – Chuẩn bị Database (MySQL)

```sql
CREATE DATABASE rice_disease_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Cập nhật `src/main/resources/application.properties`:
```properties
spring.datasource.username=root
spring.datasource.password=<mật khẩu của bạn>
```

### Bước 2 – Chạy Python AI Server

```bash
cd ai-server

# Tạo virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt

# Đặt file weights vào thư mục weights/
mkdir -p weights
# copy yolov12_biformer_p2_best.pt vào weights/

# Chạy server (port 8000)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> **Lưu ý:** Nếu chưa có file weights, server tự động chạy ở **DEMO mode** (trả kết quả ngẫu nhiên để test UI).

Kiểm tra: http://localhost:8000/health

### Bước 3 – Chạy Spring Boot Backend

Mở terminal ở thư mục gốc `giaodien_doan` và chạy:

```bash
# Build và chạy
mvn spring-boot:run
```

Kiểm tra: http://localhost:8080/api/health

### Bước 4 – Mở Giao diện Frontend

Truy cập http://localhost:8080 để mở giao diện web chẩn đoán bệnh lá lúa dạng Glassmorphism.

---

## 🔌 API Endpoints

| Method | URL | Mô tả |
|--------|-----|-------|
| `POST` | `/api/diagnose` | Upload ảnh, trả kết quả chẩn đoán |
| `GET`  | `/api/history?page=0&size=10` | Lịch sử chẩn đoán (phân trang) |
| `GET`  | `/api/history?label=Brown+Spot` | Lọc theo tên bệnh |
| `GET`  | `/api/health` | Health check |

### Ví dụ gọi `/api/diagnose`

```bash
curl -X POST http://localhost:8080/api/diagnose \
  -F "image=@/path/to/rice_leaf.jpg"
```

Response:
```json
{
  "id": 1,
  "label": "Brown Spot",
  "vietnameseName": "Đốm Nâu (Brown Spot)",
  "classIndex": 3,
  "confidence": 0.8734,
  "bboxes": [{"x": 120, "y": 80, "width": 200, "height": 150, "confidence": 0.87}],
  "annotatedImageBase64": "...(base64 JPEG)...",
  "treatment": "Bón phân kali đầy đủ. Phun Mancozeb hoặc Propiconazole định kỳ 7-10 ngày/lần.",
  "createdAt": "2025-07-10T14:30:00"
}
```

---

## 🌿 5 Classes bệnh lúa

| Index | Tên Anh | Tên Việt | Mức độ |
|-------|---------|----------|--------|
| 0 | Leaf Blast | Đạo Ôn | 🔴 Cao |
| 1 | Bacterial Leaf Blight | Bạc Lá | 🔴 Cao |
| 2 | Sheath Blight | Khô Vằn | 🟡 Trung bình |
| 3 | Brown Spot | Đốm Nâu | 🟡 Trung bình |
| 4 | Healthy | Lá Khỏe | 🟢 Khỏe |

---

## ⚙️ Kiến trúc Model AI

- **Backbone:** YOLOv12 với R-ELAN + Area Attention (A²)
- **Neck:** BiFormer Blocks (Bi-level Routing Attention) thay thế ELAN truyền thống
- **Head:** 4-scale Detect [P2, P3, P4, P5] – tối ưu phát hiện vết bệnh siêu nhỏ
- **Dataset:** RiceLeaf_5Classes_Balanced (cân bằng 5 lớp)

---

## 📝 Ghi chú phát triển

- Thay `allow_origins=["*"]` trong `WebConfig.java` bằng domain thực khi production
- Cấu hình `ai.server.url` trong `application.properties` nếu Python server chạy trên máy khác
- Frontend `index.html` tự động fallback sang Demo mode nếu Spring Boot chưa chạy
- Bảng `diagnosis_records` được tạo tự động bởi `spring.jpa.hibernate.ddl-auto=update`
