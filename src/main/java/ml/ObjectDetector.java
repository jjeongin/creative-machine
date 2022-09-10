// Use this package name when debugging from IntelliJ
package ml;

import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import processing.core.PApplet;
import processing.core.PImage;

import ml.translator.ObjectDetectorTranslator;
import ml.util.ProcessingUtils;
import ml.util.DJLUtils;
import ml.MLObject;

/**
 * Object Detector using Deep Java Library
 * @example ObjectDetectorDJLExample
 *
 */
public class ObjectDetector {
    PApplet parent; // reference to the parent sketch
    private Predictor<Image, DetectedObjects> predictor; // predictor

    private static final Logger logger =
            LoggerFactory.getLogger(ml.ObjectDetector.class);

    /**
     * constructor
     * @param myParent
     * @param modelNameOrURL : model name to load (choose from object detection models in tf model zoo)
     *                         if not in the model zoo, try to load as URL
     */
    public ObjectDetector(PApplet myParent, String modelName) {
        this.parent = myParent;
        logger.info("model loading..");
        // set a criteria to select a model to use
        Criteria<Image, DetectedObjects> criteria; // criteria for selecting the model
        // SSD trained on open images dataset from TensorFlow engine
        if (modelName.equals("openimages_ssd")) {
            criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optFilter("backbone", "mobilenet_v2")  // mobilenet has same accuracy as vgg while it is 32 times smaller than vgg
                    .optEngine("TensorFlow")
                    .build();
        }
        // SSD trained on coco dataset from MXNet engine
        else if (modelName.equals("coco_ssd")) {
            criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optFilter("backbone", "vgg16")
                    .optFilter("dataset", "coco")
                    .optEngine("MXNet")
                    .build();
        }
        // SSD trained on voc dataset from MXNet engine
        else if (modelName.equals("voc_ssd")) {
            criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optFilter("backbone", "mobilenet1.0")
                    .optFilter("dataset", "voc")
                    .optFilter("size", "512")
                    .optEngine("MXNet")
                    .build();
        }
        // Yolo trained on voc dataset from MXNet engine
        else if (modelName.equals("voc_yolo")) {
            criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optFilter("backbone", "mobilenet1.0")
                    .optFilter("dataset", "voc")
                    .optEngine("MXNet")
                    .build();
        }
        // Yolo trained on coco dataset from MXNet engine
        else if (modelName.equals("coco_yolo")) {
            criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optFilter("backbone", "mobilenet1.0")
                    .optFilter("dataset", "coco")
                    .optEngine("MXNet")
                    .build();
        }
        else {
            throw new IllegalArgumentException("No model named \'" + modelName + "\'. Check http://jjeongin.github.io/creative-machine/reference/object-detector for available model options.");
            // load a custom model with URL
            // if user passed a remote URL or local file path
//          // check if the URL is valid (source: https://stackoverflow.com/questions/2230676/how-to-check-for-a-valid-url-in-java)
//            URL url = null; // check for the URL protocol
//            try {
//                url = new URL(modelNameOrURL);
//            } catch (MalformedURLException e) {
//                throw new RuntimeException(e);
//            }
//            try {
//                url.toURI(); // extra check if the URI is valid
//            } catch (URISyntaxException e) {
//                throw new RuntimeException(e);
//            }
//            criteria = Criteria.builder()
//                    .optApplication(Application.CV.OBJECT_DETECTION)
//                    .setTypes(Image.class, DetectedObjects.class)
//                    .optModelUrls(String.valueOf(url))
//                    // saved_model.pb file is in the subfolder of the model archive file
//                    .optModelName(ProcessingUtils.getFileNameFromPath(modelNameOrURL)+"/saved_model")
//                    .optTranslator(new ObjectDetectorTranslator())
//                    .optEngine("TensorFlow")
//                    .build();
        }
        // load the model
        ZooModel<Image, DetectedObjects> model = null;
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

    // HELPER METHODS --------------------------------------------------
    /**
     * Convert each object in DetectedObjects to MLObject
     * @param detected, originalImgWidth, originalImgHeight
     * @return MLObject[]
     */
    private MLObject[] DetectedObjectsToMLObjects(DetectedObjects detected, int originalImgWidth, int originalImgHeight) {
        int numObjects = detected.getNumberOfObjects();
        MLObject[] objectList = new MLObject[numObjects];
        for (int i = 0; i < numObjects; i++) {
            // get the ith detected object
            DetectedObjects.DetectedObject d = detected.item(i);
            String className = d.getClassName(); // get class name
            float probability = (float) d.getProbability(); // get probability
            // get bounding box
            Rectangle bound = d.getBoundingBox().getBounds();
            float x = (float) bound.getX() * originalImgWidth; // get upper left corner of the bounding box
            float y = (float) bound.getY() * originalImgHeight;
            //  PVector upperLeft = new PVector((float) bound.getX(), (float) bound.getY());
            float width = (float) bound.getWidth() * originalImgWidth; // get width of the bounding box
            float height = (float) bound.getHeight() * originalImgHeight; // get height of the bounding box
            // convert each object as MLObject
            objectList[i] = new MLObject(className, probability, x, y, width, height);
        }
        return objectList;
    }

    // Experimental WIP function
    // To implement a custom saveBoundingBoxImage function instead of passing it as an option in .predict()
    /**
     * Convert objects in MLObject[] to DetectedObjects
     * @param objectList, originalImgWidth, originalImgHeight
     * @return MLObject[]
     */
    private DetectedObjects MLObjectsToDetectedObjects(MLObject[] objectList, int originalImgWidth, int originalImgHeight) {
        int numObjects = objectList.length;
        List<String> classNames = new ArrayList<>();
        List<Double> probabilities = new ArrayList<>();
        List<BoundingBox> boundingBoxes = new ArrayList<>();
        for (int i = 0; i < numObjects; i++) {
            MLObject object = objectList[i];
            classNames.add(object.getLabel());
            probabilities.add((double) object.getConfidence());
            // To do : add each bounding boxes
            // boundingBoxes.add()
        }
        DetectedObjects objects = new DetectedObjects(classNames, probabilities, boundingBoxes);
        return objects;
    }
    // --------------------------------------------------
    /**
     * Run object detection on given PImage
     * @param pImg
     * @return MLObject[]
     */
    public MLObject[] predict(PImage pImg) {
        // get original image size
        int originalImgWidth = pImg.width;
        int originalImgHeight = pImg.height;
        // convert PImage to DJL Image
        BufferedImage buffImg = ProcessingUtils.PImageToBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);
        // detect objects
        DetectedObjects objects = null;
        try {
            objects = this.predictor.predict(img);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        // parse DetectedObjects to a list of MLObject
        MLObject[] objectList = DetectedObjectsToMLObjects(objects, originalImgWidth, originalImgHeight);
        return objectList;
    }
    /**
     * Run object detection on given PImage and save output image
     * @param pImg, saveOutputImg, fileName
     * @return MLObject[]
     */
    public MLObject[] predict(PImage pImg, String fileName) {
        // get original image size
        int originalImgWidth = pImg.width;
        int originalImgHeight = pImg.height;
        // convert PImage to DJL Image
        BufferedImage buffImg = ProcessingUtils.PImageToBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);
        // detect objects
        DetectedObjects objects = null;
        try {
            objects = this.predictor.predict(img);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        // save bounding box image if file name is not null or empty
        if (fileName != null && !fileName.isEmpty() && !fileName.trim().isEmpty()) {
            DJLUtils.saveBoundingBoxImage(this.parent, fileName, img, objects);
        }
        // parse DetectedObjects to a list of MLObject
        MLObject[] objectList = DetectedObjectsToMLObjects(objects, originalImgWidth, originalImgHeight);
        return objectList;
    }
}
