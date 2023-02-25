package com.controller;

import ai.djl.MalformedModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import com.entity.Animal;
import com.service.BreedClassificationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Slf4j
@RestController
public class BreedController {

    @Autowired
    private BreedClassificationService catClassficationService;

    @PostMapping("/breed/uploading")
    public Animal upload(@RequestParam("file") MultipartFile file) throws IOException, TranslateException, ModelNotFoundException, MalformedModelException {
        return catClassficationService.getCateInfo(file);
    }


}
