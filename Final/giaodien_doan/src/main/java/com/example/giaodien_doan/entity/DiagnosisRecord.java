package com.example.giaodien_doan.entity;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "diagnosis_records")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DiagnosisRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** Tên bệnh tiếng Anh, ví dụ: "Brown Spot" */
    @Column(nullable = false)
    private String label;

    /** Tên bệnh tiếng Việt, ví dụ: "Đốm Nâu" */
    @Column(name = "vietnamese_name")
    private String vietnameseName;

    /** Index lớp: 0=Leaf Blast, 1=BLB, 2=Sheath Blight, 3=Brown Spot, 4=Healthy */
    @Column(name = "class_index")
    private Integer classIndex;

    /** Độ tin cậy từ 0.0 đến 1.0 */
    @Column(name = "confidence")
    private Double confidence;

    /** Đường dẫn file ảnh gốc lưu trên server */
    @Column(name = "image_path")
    private String imagePath;

    /** Ảnh đã vẽ bounding box, lưu dạng base64 (LONGTEXT) */
    @Column(name = "annotated_image_b64", columnDefinition = "LONGTEXT")
    private String annotatedImageBase64;

    /** Hướng dẫn xử lý bệnh */
    @Column(name = "treatment_guide", columnDefinition = "TEXT")
    private String treatmentGuide;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        this.createdAt = LocalDateTime.now();
    }
}
