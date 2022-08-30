package ml;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import processing.core.PApplet;
import processing.core.PImage;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

import ml.util.ProcessingUtils;
import ml.MLObject;

public class ImageClassifier {
    PApplet parent; // reference to the parent sketch
    private Predictor<Image, Classifications> predictor;
    private static final Logger logger =
            LoggerFactory.getLogger(ImageClassifier.class);

    public ImageClassifier(PApplet myParent, String modelNameOrURL) {
        this.parent = myParent;
        logger.info("model loading..");

        // Select a model to use
        Criteria<Image, Classifications> criteria = null;
        if (modelNameOrURL.equals("MobileNet")) {
            criteria = Criteria.builder()
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .setTypes(Image.class, Classifications.class)
                    .optFilter("flavor", "v3_small")
                    .optEngine("MXNet")
                    .build();
        }
        else if (modelNameOrURL.equals("Darknet")) {
            criteria = Criteria.builder()
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .setTypes(Image.class, Classifications.class)
                    .optFilter("layers", "53") // can we use model name directly ??
                    .optFilter("flavor", "v3")
                    .optEngine("MXNet")
                    .build();
        }

        ZooModel<Image, Classifications> model = null;
        try {
            model = criteria.loadModel();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ModelNotFoundException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }
        this.predictor = model.newPredictor();

        logger.info("successfully loaded!");
    }

    private MLObject[] ClassificationsToMLObjects(Classifications classified) {
        List<Classifications.Classification> topKClassifications = classified.topK();
        int numObjects = topKClassifications.size(); // default K is 5
        MLObject[] objectList = new MLObject[numObjects];
        for (int i = 0; i < numObjects; i++) {
            // retrieve information from a classified object
            String labelName = topKClassifications.get(i).getClassName(); // get class name
            String labelId = labelName.split(" ")[0]; // remove label id from label name
            labelName = labelName.replace(labelId + " ", "");
            float confidence = (float) topKClassifications.get(i).getProbability(); // get probability
            // add each object to the list as MLObject
            objectList[i] = new MLObject(labelName, confidence);
        }
        return objectList;
    }

    public MLObject[] classify(PImage pImg) {
        // convert PImage to DJL Image
        BufferedImage buffImg = ProcessingUtils.PImageToBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);
        // classify objects
        Classifications classified = null;
        try {
            classified = this.predictor.predict(img);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        //  parse Classifications to a list of MLObject
        MLObject[] results = ClassificationsToMLObjects(classified);
        return results;
    }
}
