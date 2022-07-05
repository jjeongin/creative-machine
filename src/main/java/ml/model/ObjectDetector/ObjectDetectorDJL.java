// Use this package name when debugging from IntelliJ
package main.java.ml.model.ObjectDetector;

import ai.djl.*;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import main.java.ml.result.MLObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import processing.core.PApplet;
import processing.core.PImage;
import processing.core.PVector;

import main.java.ml.util.ProcessingUtils;

/**
 * Object Detector using Deep Java Library
 * @example ObjectDetectorDJLExample
 *
 */
public class ObjectDetectorDJL {
    PApplet parent; // reference to the parent sketch
    private Criteria<Image, DetectedObjects> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(ObjectDetectorDJL.class);

    /**
     * constructor
     * @param myParent
     * @param modelNameOrURL : model name to load (choose from object detection models in tf model zoo)
     *                         if not in the model zoo, try to load as URL
     */
    public ObjectDetectorDJL(PApplet myParent, String modelNameOrURL) {
        this.parent = myParent;
        logger.info("model loading..");

//        String modelUrl =
//                "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz";

//        String backbone;
//        if ("TensorFlow".equals(Engine.getDefaultEngineName())) {
//            backbone = "mobilenet_v2";
//        } else {
//            backbone = "resnet50";
//        }
//        backbone = "resnet50";

        // Select the model to use
        // SSD from TensorFlow engine
        // Available models : ssd {"backbone":"mobilenet_v2","dataset":"openimages_v4"}
        if (modelNameOrURL.equals("openimages_ssd")) {
            this.criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optFilter("backbone", "mobilenet_v2")
                    .optEngine("TensorFlow")
                    .build();
        }
        // SSD from MXNet engine
        // Available models :
        // ssd_512_resnet50_v1_voc {"size":"512","backbone":"resnet50","flavor":"v1","dataset":"voc"}
        // ssd_512_vgg16_atrous_coco {"size":"512","backbone":"vgg16","flavor":"atrous","dataset":"coco"}
        // ssd_512_mobilenet1.0_voc {"size":"512","backbone":"mobilenet1.0","dataset":"voc"}
        // ssd_300_vgg16_atrous_voc {"size":"300","backbone":"vgg16","flavor":"atrous","dataset":"voc"}
        else if (modelNameOrURL.equals("cocossd")) {
            this.criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optFilter("backbone", "vgg16") // vgg has same accuracy as mobilenet while it is 32 times bigger than mobilenet
                    .optFilter("dataset", "coco")
                    .optEngine("MXNet")
                    .build();
        }
        // Yolo from MXNet engine
        // Available models :
        // dataset: "voc", "coco"
        // backbone: "darknet53", "mobilenet1.0"
        else if (modelNameOrURL.equals("yolo")) {
            this.criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optFilter("backbone", "darknet53")
                    .optFilter("dataset", "coco")
                    .optEngine("MXNet")
                    .build();
        }

        logger.info("successfully loaded!");
    }

    // HELPER METHODS --------------------------------------------------
    /**
     * Parse each object in DetectedObjects into DetectedObjectDJL[]
     * @param detected
     * @return DetectedObjectDJL[]
     */
    private MLObject[] parseDetectedObjects(DetectedObjects detected) {
        int numObjects = detected.getNumberOfObjects();
        MLObject[] objectList = new MLObject[numObjects];
        for (int i = 0; i < numObjects; i++) {
            // get the ith detected object
            DetectedObjects.DetectedObject d = detected.item(i);
            // retrieve information from a detected object
            String className = d.getClassName(); // get class name
            float probability = (float) d.getProbability(); // get probability
            Rectangle bound = d.getBoundingBox().getBounds(); // get bounding box
            PVector upperLeft = new PVector((float) bound.getX(), (float) bound.getY()); // get upper left corner of the bounding box
            float width = (float) bound.getWidth(); // get width of the bounding box
            float height = (float) bound.getHeight(); // get height of the bounding box
            // add each object to the list as DetectedObjectDJL
            objectList[i] = new MLObject(className, probability, upperLeft, width, height);
        }
        return objectList;
    }
    // --------------------------------------------------

    /**
     * Run object detection on given PImage
     * @param pImg
     * @return DetectedObjects
     */
    public MLObject[] detect(PImage pImg) {
        BufferedImage buffImg = ProcessingUtils.PImagetoBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                // detect objects
                DetectedObjects detected = predictor.predict(img);
                // parse DetectedObjects to a list of Object
                MLObject[] detectedList = parseDetectedObjects(detected);
                return detectedList;
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

    public MLObject[] detect(PImage pImg, Boolean saveOutputImg, String fileName) {
        BufferedImage buffImg = ProcessingUtils.PImagetoBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                // detect objects
                DetectedObjects detected = predictor.predict(img);
                // parse DetectedObjects to a list of Object
                MLObject[] detectedList = parseDetectedObjects(detected);
                // save bounding box image
                if (saveOutputImg == true) {
                    saveBoundingBoxImage(fileName, img, detected);
                }
                return detectedList;
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

    /**
     *
     * @param fileName
     * @param img
     * @param detected
     */
    private void saveBoundingBoxImage(String fileName, Image img, DetectedObjects detected) {
        // Default output path is parent sketch directory
        Path outputDir = Paths.get(parent.sketchPath());
        try {
            Files.createDirectories(outputDir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        img.drawBoundingBoxes(detected);

        Path imagePath = outputDir.resolve(fileName);
        // OpenJDK can't save jpg with alpha channel
        try {
            img.save(Files.newOutputStream(imagePath), "png");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Detected objects image has been saved in: " + imagePath);
    }
}
