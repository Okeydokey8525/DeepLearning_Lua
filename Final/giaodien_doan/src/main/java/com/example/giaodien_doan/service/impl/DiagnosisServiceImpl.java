package com.example.giaodien_doan.service.impl;

import com.example.giaodien_doan.dto.response.DiagnosisResultDTO;
import com.example.giaodien_doan.dto.response.PredictionResponse;
import com.example.giaodien_doan.entity.DiagnosisRecord;
import com.example.giaodien_doan.repository.DiagnosisRepository;
import com.example.giaodien_doan.service.DiagnosisService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.time.Duration;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class DiagnosisServiceImpl implements DiagnosisService {

    private final WebClient aiWebClient;
    private final DiagnosisRepository diagnosisRepository;

    // Mapping class index → tên bệnh tiếng Việt
    private static final Map<Integer, String> DISEASE_VI = Map.of(
            0, "Bạc Lá (Bacterial Leaf Blight)",
            1, "Đốm Nâu (Brown Spot)",
            2, "Đạo Ôn (Leaf Blast)",
            3, "Khô Vằn (Sheath Blight)",
            4, "Lá Khỏe (Healthy)"
    );

    // Hướng dẫn xử lý theo class index
    private static final Map<Integer, String> TREATMENT_GUIDE = Map.of(
            0, "Dùng thuốc gốc đồng (Bordeaux mixture). Bón kali cân đối, tránh bón thừa đạm. Vệ sinh đồng ruộng.",
            1, "Bón phân kali đầy đủ. Phun Mancozeb hoặc Propiconazole định kỳ 7-10 ngày/lần.",
            2, "Phun thuốc Tricyclazole hoặc Isoprothiolane. Thoát nước ruộng 3-5 ngày. Không bón thừa đạm.",
            3, "Phun Validamycin hoặc Hexaconazole. Giảm ẩm độ bằng cách tỉa thưa cây lúa.",
            4, "Lá lúa khỏe mạnh. Tiếp tục chăm sóc theo quy trình hiện tại, theo dõi định kỳ."
    );

    @Override
    public DiagnosisResultDTO diagnose(MultipartFile imageFile, String modelType) {
        log.info("Bắt đầu chẩn đoán: file={}, size={}KB, modelType={}",
                imageFile.getOriginalFilename(), imageFile.getSize() / 1024, modelType);

        try {
            // 1. Đọc bytes từ MultipartFile
            byte[] fileBytes = imageFile.getBytes();
            String originalFilename = imageFile.getOriginalFilename();

            // 2. Chuẩn bị MultipartBody để gửi qua WebClient
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            ByteArrayResource fileResource = new ByteArrayResource(fileBytes) {
                @Override
                public String getFilename() {
                    return originalFilename != null ? originalFilename : "image.jpg";
                }
            };
            body.add("file", fileResource);
            body.add("model_type", modelType);

            // 3. Gọi Python FastAPI Inference Server
            PredictionResponse prediction = aiWebClient.post()
                    .uri("/predict")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(body))
                    .retrieve()
                    .onStatus(
                            status -> status.is4xxClientError() || status.is5xxServerError(),
                            response -> response.bodyToMono(String.class)
                                     .flatMap(err -> Mono.error(
                                             new RuntimeException("AI Server lỗi: " + err)))
                    )
                    .bodyToMono(PredictionResponse.class)
                    .timeout(Duration.ofSeconds(60)) // Timeout 60 giây cho inference
                    .block();

            if (prediction == null) {
                throw new RuntimeException("Không nhận được phản hồi từ AI Server");
            }

            log.info("Kết quả inference: label={}, confidence={:.2f}, imageUrl={}",
                    prediction.getLabel(), prediction.getConfidence(), prediction.getImageUrl());

            // 4. Lưu kết quả vào database (Đảm bảo các trường bắt buộc không bị null tránh lỗi DB)
            String label = prediction.getLabel();
            if (label == null || label.trim().isEmpty()) {
                label = "Unknown";
            }

            Integer classIndex = prediction.getClassIndex();
            if (classIndex == null) {
                classIndex = 4; // Mặc định là 4 (Healthy/Lá khỏe) nếu không có dự đoán hợp lệ
            }

            Double confidence = prediction.getConfidence();
            if (confidence == null) {
                confidence = 0.0;
            }

            String viName = DISEASE_VI.getOrDefault(classIndex, label);
            String treatment = TREATMENT_GUIDE.getOrDefault(classIndex, "Tham khảo cán bộ nông nghiệp.");
            String annotatedImg = prediction.getAnnotatedImage();
            if (annotatedImg == null) {
                annotatedImg = "";
            }
            String imageUrl = prediction.getImageUrl();
            if (imageUrl == null) {
                imageUrl = "";
            }

            DiagnosisRecord result = DiagnosisRecord.builder()
                    .label(label)
                    .vietnameseName(viName)
                    .classIndex(classIndex)
                    .confidence(confidence)
                    .imagePath(imageUrl)
                    .annotatedImageBase64(annotatedImg)
                    .treatmentGuide(treatment)
                    .build();

            // In log chi tiết thông tin đối tượng trước khi thực hiện lưu DB (để phát hiện lỗi nếu có)
            log.info("[DATABASE SAVE] Đang chuẩn bị lưu bản ghi DiagnosisRecord vào cơ sở dữ liệu. Chi tiết: label='{}', vietnameseName='{}', classIndex={}, confidence={}, annotatedImageBase64Length={}, treatmentGuide='{}'",
                    result.getLabel(), result.getVietnameseName(), result.getClassIndex(), result.getConfidence(),
                    (result.getAnnotatedImageBase64() != null ? result.getAnnotatedImageBase64().length() : 0),
                    result.getTreatmentGuide());

            DiagnosisRecord savedResult = null;
            if (result != null) {
                savedResult = diagnosisRepository.save(result);
            }
            log.info("[DATABASE SAVE] Lưu cơ sở dữ liệu thành công! ID bản ghi mới tạo = {}", 
                    savedResult != null ? savedResult.getId() : null);

            // 5. Trả về DTO cho Frontend
            return DiagnosisResultDTO.builder()
                    .id(savedResult != null ? savedResult.getId() : null)
                    .label(label)
                    .vietnameseName(viName)
                    .classIndex(classIndex)
                    .confidence(confidence)
                    .bboxes(prediction.getBboxes())
                    .annotatedImageBase64(annotatedImg)
                    .imageUrl(imageUrl)
                    .treatment(treatment)
                    .createdAt(savedResult != null ? savedResult.getCreatedAt() : null)
                    .build();

        } catch (IOException e) {
            log.error("Lỗi đọc file ảnh: {}", e.getMessage());
            throw new RuntimeException("Không thể đọc file ảnh tải lên", e);
        }
    }

    @Override
    public Page<DiagnosisRecord> getHistory(int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        return diagnosisRepository.findAll(pageable);
    }

    @Override
    public Page<DiagnosisRecord> getHistoryByLabel(String label, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        return diagnosisRepository.findByLabel(label, pageable);
    }
}
