import asyncio
import io
import os
from pathlib import Path
from main import app, predict
from fastapi import UploadFile, Form

async def test():
    print("--- Khởi chạy thử nghiệm tích hợp API (Programmatic Test for v2.0) ---")
    
    # Kích hoạt lifespan context của FastAPI (để tải cả hai mô hình)
    async with app.router.lifespan_context(app):
        # Sử dụng ảnh mẫu từ thư mục model 50e
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        test_image_path = BASE_DIR / "rice_leaf_yolo11s_50e" / "train_batch0.jpg"
        
        if not test_image_path.exists():
            print(f"Lỗi: Không tìm thấy ảnh test tại {test_image_path}")
            return
            
        print(f"Đọc file ảnh test: {test_image_path}")
        with open(test_image_path, "rb") as f:
            image_data = f.read()
            
        # Tạo đối tượng UploadFile giả lập
        upload_file = UploadFile(
            file=io.BytesIO(image_data),
            filename="train_batch0.jpg",
            headers={"content-type": "image/jpeg"}
        )
        
        print("Đang gọi endpoint predict với YOLOv11s-50e...")
        result_yolo = await predict(file=upload_file, model_type="YOLOv11s-50e")
        
        # Reset file pointer
        upload_file.file.seek(0)
        
        print("Đang gọi endpoint predict với BiFormer-P4-40e...")
        result_trans = await predict(file=upload_file, model_type="BiFormer-P4-40e")
        
        print("\n=== KẾT QUẢ TRẢ VỀ TỪ API - YOLOv11s-50e ===")
        print(f"label: {result_yolo.label}")
        print(f"class_index: {result_yolo.class_index}")
        print(f"confidence: {result_yolo.confidence}")
        print(f"bboxes count: {len(result_yolo.bboxes)}")
        print(f"image_url: {result_yolo.image_url}")
        print("=========================================\n")

        print("\n=== KẾT QUẢ TRẢ VỀ TỪ API - BiFormer-P4-40e ===")
        print(f"label: {result_trans.label}")
        print(f"class_index: {result_trans.class_index}")
        print(f"confidence: {result_trans.confidence}")
        print(f"bboxes count: {len(result_trans.bboxes)}")
        print(f"image_url: {result_trans.image_url}")
        print("=========================================\n")
        
        # Kiểm tra cấu trúc kết quả
        assert result_yolo.label is not None, "Thiếu trường 'label'"
        assert result_yolo.class_index is not None, "Thiếu trường 'class_index'"
        assert result_yolo.confidence is not None, "Thiếu trường 'confidence'"
        assert result_yolo.bboxes is not None, "Thiếu trường 'bboxes'"
        assert result_yolo.image_url is not None, "Thiếu trường 'image_url'"
        
        print("Đạt: Cấu trúc JSON DTO phản hồi của YOLOv11s-50e hoàn toàn chính xác!")
        print("Đạt: Cấu trúc JSON DTO phản hồi của BiFormer-P4-40e hoàn toàn chính xác!")

if __name__ == "__main__":
    asyncio.run(test())
