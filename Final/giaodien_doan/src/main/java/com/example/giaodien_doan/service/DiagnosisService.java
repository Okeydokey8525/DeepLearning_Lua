package com.example.giaodien_doan.service;

import com.example.giaodien_doan.dto.response.DiagnosisResultDTO;
import com.example.giaodien_doan.entity.DiagnosisRecord;
import org.springframework.data.domain.Page;
import org.springframework.web.multipart.MultipartFile;

public interface DiagnosisService {

    /**
     * Nhận ảnh upload, gửi đến Python Inference Server,
     * lưu kết quả vào DB, trả về DTO cho Frontend.
     */
    DiagnosisResultDTO diagnose(MultipartFile imageFile, String modelType);

    /**
     * Lấy lịch sử chẩn đoán có phân trang, sắp xếp mới nhất trước.
     */
    Page<DiagnosisRecord> getHistory(int page, int size);

    /**
     * Lọc lịch sử theo tên bệnh.
     */
    Page<DiagnosisRecord> getHistoryByLabel(String label, int page, int size);
}
