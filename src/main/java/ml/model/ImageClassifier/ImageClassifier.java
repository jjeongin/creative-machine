package main.java.ml.model.ImageClassifier;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import main.java.ml.model.ObjectDetector.ObjectDetectorDJL;
import main.java.ml.result.MLObject;
import main.java.ml.util.ProcessingUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import processing.core.PApplet;
import processing.core.PImage;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

public class ImageClassifier {
    PApplet parent; // reference to the parent sketch
    private Criteria<Image, Classifications> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(ObjectDetectorDJL.class);

    public ImageClassifier(PApplet myParent, String modelNameOrURL) {
        this.parent = myParent;
        logger.info("model loading..");

        if (modelNameOrURL.equals("MobileNet")) {
            this.criteria = Criteria.builder()
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .setTypes(Image.class, Classifications.class)
                    .optFilter("flavor", "v3_small")
                    .optEngine("MXNet")
                    .build();
        }
        else if (modelNameOrURL.equals("Darknet")) {
            this.criteria = Criteria.builder()
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .setTypes(Image.class, Classifications.class)
                    .optFilter("layers", "53") // can we use model name directly ??
                    .optFilter("flavor", "v3")
                    .optEngine("MXNet")
                    .build();
        }

        logger.info("successfully loaded!");
    }

    private MLObject[] parseClassifications(Classifications classified) {
        List<Classifications.Classification> topKClassifications = classified.topK();
        int numObjects = topKClassifications.size(); // default K is 5
        MLObject[] objectList = new MLObject[numObjects];
        for (int i = 0; i < numObjects; i++) {
            // retrieve information from a detected object
            String labelName = topKClassifications.get(i).getClassName(); // get class name
            String labelId = labelName.split(" ")[0]; // remove label id from label name
            labelName = labelName.replace(labelId + " ", "");
            float confidence = (float) topKClassifications.get(i).getProbability(); // get probability
            // add each object to the list as DetectedObjectDJL
            objectList[i] = new MLObject(labelName, confidence);
        }
        return objectList;
    }

    public MLObject[] classify(PImage pImg) {
        BufferedImage buffImg = ProcessingUtils.PImagetoBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);

        try (ZooModel<Image, Classifications> model = criteria.loadModel()) {
            try (Predictor<Image, Classifications> predictor = model.newPredictor()) {
                // classify objects
                Classifications classified = predictor.predict(img);
                // parse Classifications to a list of Object
                MLObject[] results = parseClassifications(classified);
                return results;
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }
        } catch (ModelNotFoundException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
