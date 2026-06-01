package com.example.giaodien_doan.controller;

import com.example.giaodien_doan.dto.response.DiagnosisResultDTO;
import com.example.giaodien_doan.entity.DiagnosisRecord;
import com.example.giaodien_doan.service.DiagnosisService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

/**
 * REST Controller xử lý chẩn đoán bệnh lá lúa.
 *
 * Endpoints:
 *   POST /api/diagnose        - Upload ảnh và nhận kết quả chẩn đoán
 *   GET  /api/history         - Lấy lịch sử chẩn đoán (phân trang)
 *   GET  /api/health          - Health check
 */
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@Slf4j
public class DiagnosisController {

    private final DiagnosisService diagnosisService;

    /**
     * Endpoint chính: nhận ảnh upload và trả về kết quả chẩn đoán.
     * POST /api/diagnose
     * Content-Type: multipart/form-data
     * Form field: "image" (file)
     */
    @PostMapping(value = "/diagnose", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> diagnose(
            @RequestParam("image") MultipartFile imageFile,
            @RequestParam(value = "modelType", defaultValue = "YOLOv11s-50e") String modelType) {

        // Validation
        if (imageFile == null || imageFile.isEmpty()) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "Vui lòng chọn file ảnh"));
        }

        String contentType = imageFile.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            return ResponseEntity.status(HttpStatus.UNSUPPORTED_MEDIA_TYPE)
                    .body(Map.of("error", "Chỉ chấp nhận file ảnh (JPG, PNG, WEBP)"));
        }

        if (imageFile.getSize() > 10 * 1024 * 1024) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "File ảnh không được vượt quá 10MB"));
        }

        log.info("Nhận yêu cầu chẩn đoán: filename={}, modelType={}", imageFile.getOriginalFilename(), modelType);

        try {
            DiagnosisResultDTO result = diagnosisService.diagnose(imageFile, modelType);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            log.error("Lỗi khi chẩn đoán: {}", e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Lỗi server: " + e.getMessage()));
        }
    }

    /**
     * Lấy lịch sử chẩn đoán, có phân trang và lọc theo bệnh.
     * GET /api/history?page=0&size=10&label=Brown+Spot
     */
    @GetMapping("/history")
    public ResponseEntity<Page<DiagnosisRecord>> getHistory(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(required = false) String label) {

        Page<DiagnosisRecord> history;
        if (label != null && !label.isBlank()) {
            history = diagnosisService.getHistoryByLabel(label, page, size);
        } else {
            history = diagnosisService.getHistory(page, size);
        }
        return ResponseEntity.ok(history);
    }

    /**
     * Health check endpoint.
     * GET /api/health
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of(
                "status", "UP",
                "service", "RiceGuard AI Backend",
                "version", "1.0.0"
        ));
    }
}
