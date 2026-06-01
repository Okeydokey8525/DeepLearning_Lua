package com.example.giaodien_doan.dto.response;

import lombok.Builder;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

/**
 * DTO trả về cho Frontend sau khi chẩn đoán thành công.
 */
@Data
@Builder
public class DiagnosisResultDTO {

    private Long id;
    private String label;
    private String vietnameseName;
    private Integer classIndex;
    private Double confidence;

    /** Danh sách bounding boxes từ model */
    private List<PredictionResponse.BBoxDTO> bboxes;

    /** Ảnh đã vẽ bbox dạng base64 để hiển thị trực tiếp trên <img> tag */
    private String annotatedImageBase64;

    /** Đường dẫn URL tương đối đến ảnh có vẽ bbox */
    private String imageUrl;

    /** Hướng dẫn xử lý bệnh tương ứng */
    private String treatment;

    private LocalDateTime createdAt;
}
