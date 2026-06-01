package com.example.giaodien_doan.dto.response;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.List;

/**
 * DTO nhận JSON phản hồi từ Python FastAPI Inference Server.
 * Tương ứng với cấu trúc: { label, class_index, confidence, bboxes, annotated_image }
 */
@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class PredictionResponse {

    /** Tên bệnh: "Leaf Blast", "Brown Spot", v.v. */
    private String label;

    /** Index lớp: 0-4 */
    @JsonProperty("class_index")
    private Integer classIndex;

    /** Độ tin cậy 0.0 - 1.0 */
    private Double confidence;

    /** Danh sách bounding boxes */
    private List<BBoxDTO> bboxes;

    /** Ảnh đã vẽ bbox, encoded base64 */
    @JsonProperty("annotated_image")
    private String annotatedImage;

    /** Đường dẫn URL tương đối đến ảnh có vẽ bbox */
    @JsonProperty("image_url")
    private String imageUrl;

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class BBoxDTO {
        private Double x;
        private Double y;
        private Double width;
        private Double height;
        private Double confidence;
    }
}
