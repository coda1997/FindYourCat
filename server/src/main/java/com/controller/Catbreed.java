package com.controller;

import com.service.CatBreedClassficationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Slf4j
@RestController
@ResponseBody
public class Catbreed {

    @Autowired
    private CatBreedClassficationService catClassficationService;

    @PostMapping("/catbreed/uploading")
    public String upload(@RequestParam("file") MultipartFile file) throws IOException {
        catClassficationService.transferFile(file);

        return catClassficationService.getCatInfo(file);
    }

}
