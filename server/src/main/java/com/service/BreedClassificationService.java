package com.service;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.*;
import com.entity.Animal;
import com.entity.Category;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.file.Paths;

@Slf4j
@Service
public class BreedClassificationService {

    public Animal getCateInfo(MultipartFile multipartFile) throws MalformedModelException, IOException, ModelNotFoundException, TranslateException {
        Category[] categories=Category.values();

        InputStream inputStream=multipartFile.getInputStream();

        Translator<Image,String> translator=new Translator<Image, String>() {

            @Override
            public Batchifier getBatchifier() {
                return Batchifier.STACK;
            }

            @Override
            public String processOutput(TranslatorContext translatorContext, NDList ndList) {
                int index=ndList.get(0).argMax().toType(DataType.INT32, false).getInt();
                return index+"";
            }

            @Override
            public NDList processInput(TranslatorContext ctx, Image image) {
                NDArray ndArray=image.toNDArray(ctx.getNDManager());

                Resize resize=new Resize(224,224);
                ndArray=resize.transform(ndArray);

                ndArray = new ToTensor().transform(ndArray);

                return new NDList(ndArray);
            }
        };

        Criteria<Image,String> criteria=Criteria.builder()
                .setTypes(Image.class,String.class)
                .optModelPath(Paths.get("model/vgg16_model.pt"))
                .optOption("mapLocation","cpu")
                .optTranslator(translator)
                .build();

        ZooModel<Image,String> model=criteria.loadModel();
        Predictor<Image,String> predictor=model.newPredictor();

        Image img=ImageFactory.getInstance().fromInputStream(inputStream);

        String res= predictor.predict(img);
        int index=Integer.parseInt(res);

        String breed=categories[index].toString();
        Animal animal=new Animal();
        animal.setBreed(breed);

        return animal;
    }
}
