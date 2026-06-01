package com.example.giaodien_doan.repository;

import com.example.giaodien_doan.entity.DiagnosisRecord;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface DiagnosisRepository extends JpaRepository<DiagnosisRecord, Long> {

    /** Lọc theo tên bệnh (có phân trang) */
    Page<DiagnosisRecord> findByLabel(String label, Pageable pageable);

    /** Lọc theo class index */
    Page<DiagnosisRecord> findByClassIndex(Integer classIndex, Pageable pageable);

    /** Lấy theo khoảng thời gian */
    List<DiagnosisRecord> findByCreatedAtBetween(LocalDateTime from, LocalDateTime to);

    /** Đếm số lần phát hiện theo bệnh */
    long countByLabel(String label);
}
