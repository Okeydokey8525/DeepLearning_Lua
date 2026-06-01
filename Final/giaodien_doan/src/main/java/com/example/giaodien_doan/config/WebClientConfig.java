package com.example.giaodien_doan.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {

    @Value("${ai.server.url:http://localhost:8000}")
    private String aiServerUrl;

    /**
     * Bean WebClient để gọi Python FastAPI Inference Server.
     * Buffer tối đa 50MB để xử lý ảnh có độ phân giải cao.
     */
    @Bean
    public WebClient aiWebClient() {
        return WebClient.builder()
                .baseUrl(aiServerUrl)
                .codecs(config -> config
                        .defaultCodecs()
                        .maxInMemorySize(50 * 1024 * 1024)) // 50MB
                .defaultHeader(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }
}
