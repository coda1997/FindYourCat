package com.service;

import com.alibaba.fastjson.JSONObject;
import com.entity.Animal;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@Slf4j
@Service
public class CatBreedClassficationService {
    @Autowired
    private RestTemplate restTemplate;

    private final String filePath= "Todo";

    public void transferFile(MultipartFile file) throws IOException {
        File file1=new File(filePath);
        if(!file1.exists()){
            boolean isSuccess=file1.mkdirs();
            if(!isSuccess){
                log.error("mkdirs failed");
            }
        }

        String filename=file.getOriginalFilename();
        String newFilePath=filePath+filename;
        file.transferTo(new File(newFilePath));
    }

    public Animal getCatInfo(MultipartFile file) {
        String uploadApi = "Todo";

        String filename = file.getOriginalFilename();

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> map = new LinkedMultiValueMap<>();
        map.add("file", new FileSystemResource(filePath + filename));

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(map, headers);

        return restTemplate.postForObject(uploadApi, request, Animal.class);
    }
}
