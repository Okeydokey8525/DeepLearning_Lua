"""
RiceGuard AI - Python FastAPI Inference Server
Model: YOLOv11s-50e & BiFormer-P4-40e
Classes: 0=Leaf Blast, 1=Bacterial Leaf Blight, 2=Sheath Blight, 3=Brown Spot, 4=Healthy

Yêu cầu:
    pip install fastapi uvicorn ultralytics pillow python-multipart torch torchvision opencv-python-headless

Chạy server:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import base64
import logging
import uuid
import sys
import types
from pathlib import Path
from typing import List, Optional

import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ── Đăng ký động module BiFormer để Ultralytics có thể load được model ──
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class BiLevelRoutingAttention(nn.Module):
    def __init__(self, c2):
        super().__init__()
        self.routing_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 4, 1),
            nn.ReLU(),
            nn.Conv2d(c2 // 4, c2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.routing_layer(x)

class BiFormer(nn.Module):
    def __init__(self, c1, c2=None, n=1):
        super().__init__()
        if c2 is None:
            c2 = c1
        self.conv = Conv(c1, c2, 3)
        self.norm = nn.LayerNorm(c2)
        self.attn = BiLevelRoutingAttention(c2)

    def forward(self, x):
        x = self.conv(x)
        # Permute for LayerNorm: (batch, channels, H, W) -> (batch, H, W, channels)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Permute back: (batch, H, W, channels) -> (batch, channels, H, W)
        x = x.permute(0, 3, 1, 2)
        return self.attn(x)

# Tạo module giả lập và đưa vào sys.modules
biformer_module = types.ModuleType("ultralytics.nn.modules.biformer")
biformer_module.BiFormer = BiFormer
biformer_module.BiLevelRoutingAttention = BiLevelRoutingAttention
biformer_module.Conv = Conv
sys.modules["ultralytics.nn.modules.biformer"] = biformer_module

# Thêm vào namespace của ultralytics để parse model cấu hình
import ultralytics.nn.tasks
import ultralytics.nn.modules
setattr(ultralytics.nn.modules, "BiFormer", BiFormer)
setattr(ultralytics.nn.modules, "BiLevelRoutingAttention", BiLevelRoutingAttention)
setattr(ultralytics.nn.tasks, "BiFormer", BiFormer)
setattr(ultralytics.nn.tasks, "BiLevelRoutingAttention", BiLevelRoutingAttention)

# ── Cấu hình logging ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Khởi tạo FastAPI ──────────────────────────────────────────
app = FastAPI(
    title="RiceGuard AI Inference Server",
    description="YOLOv11s-50e and BiFormer-P4-40e rice leaf disease detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Production: thay bằng domain Spring Boot
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ────────────────────────────────────────────────
CLASS_NAMES = [
    "Leaf Blast",
    "Bacterial Leaf Blight",
    "Sheath Blight",
    "Brown Spot",
    "Healthy"
]

# Màu vẽ bounding box cho từng lớp (RGB)
BBOX_COLORS = [
    (248, 113, 113),   # Leaf Blast     → đỏ nhạt
    (251, 146,  60),   # BLB            → cam
    (250, 204,  21),   # Sheath Blight  → vàng
    (167, 139, 250),   # Brown Spot     → tím
    ( 74, 222, 128),   # Healthy        → xanh lá
]

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 640

# Xác định đường dẫn tương đối dựa trên vị trí tệp tin main.py (gốc TH_DeepLearning)
current_file_path = Path(__file__).resolve()
if current_file_path.parent.name == "ai-server":
    BASE_DIR = current_file_path.parent.parent.parent
else:
    BASE_DIR = current_file_path.parent

YOLO_MODEL_PATH = BASE_DIR / "rice_leaf_yolo11s_50e" / "weights" / "best.pt"
TRANSFORMER_MODEL_PATH = BASE_DIR / "rice_leaf_yolov11s_50e_biformer_P4_40e" / "weights" / "best.pt"

# ── Load model (chạy 1 lần lúc khởi động) ────────────────────
models = {}

@app.on_event("startup")
async def load_models():
    global models
    from ultralytics import YOLO
    
    # Load YOLOv11s-50e
    if YOLO_MODEL_PATH.exists():
        try:
            models["yolo"] = YOLO(str(YOLO_MODEL_PATH))
            logger.info(f"✅ Đã load YOLOv11s-50e từ {YOLO_MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ Lỗi load YOLO model: {e}")
    else:
        logger.warning(f"⚠️ Không tìm thấy weights YOLO tại {YOLO_MODEL_PATH}")

    # Load Transformer (BiFormer-P4-40e)
    if TRANSFORMER_MODEL_PATH.exists():
        try:
            models["transformer"] = YOLO(str(TRANSFORMER_MODEL_PATH))
            logger.info(f"✅ Đã load BiFormer-P4-40e từ {TRANSFORMER_MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ Lỗi load Transformer model: {e}")
    else:
        logger.warning(f"⚠️ Không tìm thấy weights Transformer tại {TRANSFORMER_MODEL_PATH}")

# ── Pydantic Response Schemas ─────────────────────────────────
class BBoxDTO(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float

class PredictResponse(BaseModel):
    label: str
    class_index: int
    confidence: float
    bboxes: List[BBoxDTO]
    annotated_image: Optional[str] = None  # base64 JPEG
    image_url: Optional[str] = None  # URL tĩnh để hiển thị trực tiếp

# ── Helper: vẽ bounding boxes lên ảnh bằng OpenCV ────────────
def draw_bboxes(image: Image.Image, detections: list) -> Image.Image:
    # Chuyển PIL Image thành OpenCV BGR image
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for det in detections:
        cls_idx = det["class_index"]
        conf    = det["confidence"]
        x1, y1  = det["x1"], det["y1"]
        x2, y2  = det["x2"], det["y2"]
        
        # Lấy màu theo định dạng RGB và chuyển thành BGR cho OpenCV
        color_rgb = BBOX_COLORS[cls_idx % len(BBOX_COLORS)]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        label = f"{CLASS_NAMES[cls_idx]} {conf:.0%}"

        # Vẽ hình chữ nhật bounding box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color_bgr, 3)

        # Cài đặt font chữ và kích thước
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Lấy kích thước text để vẽ background phía sau chữ
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Tính toán vị trí nhãn, nếu nhãn bị vượt quá mép trên cùng thì hiển thị ở dưới
        label_y1 = y1 - text_h - 10 if y1 - text_h - 10 > 0 else y1
        label_y2 = y1 if y1 - text_h - 10 > 0 else y1 + text_h + 10
        
        # Vẽ background cho text
        cv2.rectangle(img_cv, (x1, label_y1), (x1 + text_w + 10, label_y2), color_bgr, -1)
        
        # Chọn màu chữ tương phản (chữ trắng cho hầu hết các màu, chữ đen cho nền màu sáng)
        text_color = (0, 0, 0) if cls_idx == 2 else (255, 255, 255)
        
        # Ghi chữ lên ảnh
        cv2.putText(
            img_cv,
            label,
            (x1 + 5, y1 - 5 if y1 - text_h - 10 > 0 else y1 + text_h + 5),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA
        )

    # Chuyển đổi ngược từ BGR sang PIL Image (RGB)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def encode_image_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── DEMO mode: kết quả ngẫu nhiên khi chưa có weights ────────
def get_demo_result(image: Image.Image) -> PredictResponse:
    import random
    cls_idx = random.randint(0, 4)
    conf    = round(random.uniform(0.75, 0.97), 4)
    w, h    = image.size
    x1 = int(w * 0.2); y1 = int(h * 0.2)
    x2 = int(w * 0.75); y2 = int(h * 0.75)

    demo_dets = [{"class_index": cls_idx, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}]
    annotated = draw_bboxes(image.copy(), demo_dets)
    
    # Lưu ảnh vẽ khung dạng file tĩnh
    filename = f"predicted_{uuid.uuid4().hex}.jpg"
    src_upload_dir = BASE_DIR / "giaodien_doan" / "src" / "main" / "resources" / "static" / "uploads"
    src_upload_dir.mkdir(parents=True, exist_ok=True)
    src_path = src_upload_dir / filename
    annotated.save(str(src_path), format="JPEG", quality=90)
    
    target_upload_dir = BASE_DIR / "giaodien_doan" / "target" / "classes" / "static" / "uploads"
    if target_upload_dir.exists():
        target_path = target_upload_dir / filename
        annotated.save(str(target_path), format="JPEG", quality=90)
        logger.info(f"[DEMO] Saved annotated image to target/classes: {target_path}")

    return PredictResponse(
        label=CLASS_NAMES[cls_idx],
        class_index=cls_idx,
        confidence=conf,
        bboxes=[BBoxDTO(x=x1, y=y1, width=x2-x1, height=y2-y1, confidence=conf)],
        annotated_image=encode_image_base64(annotated),
        image_url=f"/uploads/{filename}"
    )

# ── Main Predict Endpoint ─────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form("YOLOv11s-50e")
):
    """
    Nhận ảnh lá lúa, chọn model dựa vào model_type và thực hiện inference.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể đọc file ảnh")

    logger.info(f"Inference: {file.filename} ({image.size[0]}x{image.size[1]}) | model_type={model_type}")

    # Lựa chọn model dựa vào model_type
    selected_model = None
    m_type = model_type.lower()
    if "transformer" in m_type or "biformer" in m_type:
        selected_model = models.get("transformer")
        model_name_used = "BiFormer-P4-40e"
    else:
        selected_model = models.get("yolo")
        model_name_used = "YOLOv11s-50e"

    # Fallback nếu model được chọn chưa được load
    if selected_model is None:
        if "yolo" in models:
            selected_model = models["yolo"]
            model_name_used = "YOLOv11s-50e (Fallback)"
        elif "transformer" in models:
            selected_model = models["transformer"]
            model_name_used = "BiFormer-P4-40e (Fallback)"

    # Demo mode nếu không có model nào được load thành công
    if selected_model is None:
        logger.warning("Không có model nào được load. Chạy DEMO mode.")
        return get_demo_result(image)

    logger.info(f"Sử dụng mô hình: {model_name_used}")

    # ── Real inference ──────────────────────────────────────
    results = selected_model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )

    result = results[0]
    boxes  = result.boxes

    filename = f"predicted_{uuid.uuid4().hex}.jpg"
    
    if boxes is None or len(boxes) == 0:
        # Không phát hiện vết bệnh → trả về Healthy
        # Lưu ảnh gốc đã copy để trả về dạng URL tĩnh
        src_upload_dir = BASE_DIR / "giaodien_doan" / "src" / "main" / "resources" / "static" / "uploads"
        src_upload_dir.mkdir(parents=True, exist_ok=True)
        src_path = src_upload_dir / filename
        image.save(str(src_path), format="JPEG", quality=90)
        
        target_upload_dir = BASE_DIR / "giaodien_doan" / "target" / "classes" / "static" / "uploads"
        if target_upload_dir.exists():
            target_path = target_upload_dir / filename
            image.save(str(target_path), format="JPEG", quality=90)

        return PredictResponse(
            label="Healthy",
            class_index=4,
            confidence=0.95,
            bboxes=[],
            annotated_image=encode_image_base64(image),
            image_url=f"/uploads/{filename}"
        )

    # Lấy detection có confidence cao nhất
    best_idx  = int(boxes.conf.argmax())
    best_cls  = int(boxes.cls[best_idx].item())
    best_conf = float(boxes.conf[best_idx].item())
    xyxy      = boxes.xyxy[best_idx].tolist()
    x1, y1, x2, y2 = [int(v) for v in xyxy]

    # Tất cả detections để vẽ
    all_dets = [
        {
            "class_index": int(boxes.cls[i].item()),
            "confidence":  float(boxes.conf[i].item()),
            "x1": int(boxes.xyxy[i][0]), "y1": int(boxes.xyxy[i][1]),
            "x2": int(boxes.xyxy[i][2]), "y2": int(boxes.xyxy[i][3]),
        }
        for i in range(len(boxes))
    ]

    annotated = draw_bboxes(image.copy(), all_dets)

    # Lưu ảnh kết quả có bounding boxes vẽ bằng OpenCV
    src_upload_dir = BASE_DIR / "giaodien_doan" / "src" / "main" / "resources" / "static" / "uploads"
    src_upload_dir.mkdir(parents=True, exist_ok=True)
    src_path = src_upload_dir / filename
    annotated.save(str(src_path), format="JPEG", quality=90)

    target_upload_dir = BASE_DIR / "giaodien_doan" / "target" / "classes" / "static" / "uploads"
    if target_upload_dir.exists():
        target_path = target_upload_dir / filename
        annotated.save(str(target_path), format="JPEG", quality=90)
        logger.info(f"Saved annotated image to target/classes: {target_path}")

    bboxes_dto = [
        BBoxDTO(
            x=d["x1"], y=d["y1"],
            width=d["x2"] - d["x1"], height=d["y2"] - d["y1"],
            confidence=d["confidence"]
        )
        for d in all_dets
    ]

    logger.info(f"Kết quả: {CLASS_NAMES[best_cls]} conf={best_conf:.3f} | tổng {len(all_dets)} box | image_url=/uploads/{filename}")

    return PredictResponse(
        label=CLASS_NAMES[best_cls],
        class_index=best_cls,
        confidence=round(best_conf, 4),
        bboxes=bboxes_dto,
        annotated_image=encode_image_base64(annotated),
        image_url=f"/uploads/{filename}"
    )

@app.get("/health")
async def health():
    return {
        "status": "UP",
        "models_loaded": {k: True for k in models.keys()},
        "yolo_model_path": str(YOLO_MODEL_PATH),
        "transformer_model_path": str(TRANSFORMER_MODEL_PATH),
        "classes": CLASS_NAMES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

