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
import ml.MLLabel;

public class ImageClassifier {
    PApplet parent; // reference to the parent sketch
    private Predictor<Image, Classifications> predictor;
    private static final Logger logger =
            LoggerFactory.getLogger(ImageClassifier.class);

    public ImageClassifier(PApplet myParent, String modelName) {
        this.parent = myParent;
        logger.info("model loading..");
        // set a criteria to select a model to use
        Criteria<Image, Classifications> criteria = null;
        if (modelName.equals("MobileNet")) {
            criteria = Criteria.builder()
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .setTypes(Image.class, Classifications.class)
                    .optFilter("flavor", "v3_small")
                    .optEngine("MXNet")
                    .build();
        }
        else if (modelName.equals("Darknet")) {
            criteria = Criteria.builder()
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .setTypes(Image.class, Classifications.class)
                    .optFilter("layers", "53")
                    .optFilter("flavor", "v3")
                    .optEngine("MXNet")
                    .build();
        }
        else {
            throw new IllegalArgumentException("No model named \'" + modelName + "\'. Check http://jjeongin.github.io/creative-machine/reference/image-classifier for available model options.");
        }
        // load the model
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
        // initialize a predictor for the model
        this.predictor = model.newPredictor();
        logger.info("successfully loaded!");
    }

    private MLLabel[] ClassificationsToMLLabels(Classifications classified) {
        List<Classifications.Classification> topKClassifications = classified.topK();
        int numObjects = topKClassifications.size(); // default K is 5
        MLLabel[] labels = new MLLabel[numObjects];
        for (int i = 0; i < numObjects; i++) {
            // retrieve information from a classified object
            String labelName = topKClassifications.get(i).getClassName(); // get class name
            String labelId = labelName.split(" ")[0]; // remove label id from label name
            labelName = labelName.replace(labelId + " ", "");
            float confidence = (float) topKClassifications.get(i).getProbability(); // get probability
            // add each object to the list as MLObject
            labels[i] = new MLLabel(labelName, confidence);
        }
        return labels;
    }

    public MLLabel[] predict(PImage pImg) {
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
        MLLabel[] results = ClassificationsToMLLabels(classified);
        return results;
    }
}
