package com.service;

import com.alibaba.fastjson.JSONObject;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@Slf4j
@Service
public class CatBreedClassficationService {

    public void transferFile(MultipartFile file) throws IOException {
        String filePath="";

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

    public String getCatInfo(MultipartFile file){
        // 调用模型得到结果

        JSONObject res=new JSONObject();

        res.put("breed","cat");
        res.put("gender","male");
        res.put("age","2");

        return res.toJSONString();
    }
}
