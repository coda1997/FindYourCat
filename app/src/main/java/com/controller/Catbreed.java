package com.controller;

import com.alibaba.fastjson.JSONObject;
import com.fasterxml.jackson.databind.util.JSONPObject;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;
import java.io.File;
import java.io.IOException;
import java.util.UUID;

@RestController
@ResponseBody
//@RequestMapping("")
public class Catbreed {
//    @Value("${file.path}")
    @PostMapping("/catbreed/uploading")
    public String upload(@RequestParam("file") MultipartFile file) throws IOException {
        String filePath="C:\\Users\\59776\\Desktop\\pic\\";

        File file1=new File(filePath);
        if(!file1.exists()){
            boolean isSuccess=file1.mkdirs();
            if(!isSuccess){
                System.out.println("mkdirs failed");
            }
        }

        String filename=file.getOriginalFilename();
//        String suffix=filename.substring(filename.lastIndexOf(".")+1);

        JSONObject res=new JSONObject();
//        if(!verifySuf(suffix)){
//            res.put("breed","error");
//        }
        String newFileName= UUID.randomUUID()+filename;
        String newFilePath=filePath+newFileName;

        file.transferTo(new File(newFilePath));

        // 调用模型得到图片识别结果

        res.put("breed","cat");
        res.put("gender","male");
        res.put("age","2");
        return res.toJSONString();
    }

//    public boolean verifySuf(String suffix){
//        if(suffix.equals("jpg")||suffix.equals("jpeg")){
//            return true;
//        }
//
//        return false;
//    }
}
