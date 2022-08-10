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
    private Criteria<Image, DetectedObjects> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(ml.ObjectDetector.class);

    /**
     * constructor
     * @param myParent
     * @param modelNameOrURL : model name to load (choose from object detection models in tf model zoo)
     *                         if not in the model zoo, try to load as URL
     */
    public ObjectDetector(PApplet myParent, String modelNameOrURL) {
        this.parent = myParent;
        logger.info("model loading..");

        // using remote url
//        String modelUrl =
//                "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz";

        // Select a model to use
        // Open SSD from TensorFlow engine
        // Available models : ssd {"backbone":"mobilenet_v2","dataset":"openimages_v4"}
        if (modelNameOrURL.equals("openimages_ssd")) {
            this.criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optFilter("backbone", "mobilenet_v2")
                    .optEngine("TensorFlow")
                    .build();
        }
        // Coco SSD from MXNet engine
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
        // if user passed remote URL or local path
        else {
            // check if the URL is valid (source: https://stackoverflow.com/questions/2230676/how-to-check-for-a-valid-url-in-java)
            URL url = null; // check for the URL protocol
            try {
                url = new URL(modelNameOrURL);
            } catch (MalformedURLException e) {
                throw new RuntimeException(e);
            }
            try {
                url.toURI(); // extra check if the URI is valid
            } catch (URISyntaxException e) {
                throw new RuntimeException(e);
            }

            this.criteria = Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION)
                    .setTypes(Image.class, DetectedObjects.class)
                    .optModelUrls(String.valueOf(url))
                    // saved_model.pb file is in the subfolder of the model archive file
                    .optModelName(ProcessingUtils.getFileNameFromPath(modelNameOrURL))
                    .optTranslator(new ObjectDetectorTranslator())
                    .optEngine("TensorFlow")
                    .build();
        }

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

    /**
     * Convert objects in MLObject[] to DetectedObjects
     * @param objectList, originalImgWidth, originalImgHeight
     * @return MLObject[]
     */
    private DetectedObjects DetectedObjectsToMLObjects(MLObject[] objectList, int originalImgWidth, int originalImgHeight) {
        int numObjects = objectList.length;
        List<String> classNames = new ArrayList<>();
        List<Double> probabilities = new ArrayList<>();
        List<BoundingBox> boundingBoxes = new ArrayList<>();
        for (int i = 0; i < numObjects; i++) {
            MLObject object = objectList[i];
            classNames.add(object.getLabel());
            probabilities.add((double) object.getConfidence());
//            boundingBoxes.add()
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
    public MLObject[] detect(PImage pImg) {
        // get original image size
        int originalImgWidth = pImg.width;
        int originalImgHeight = pImg.height;
        // convert PImage to DJL Image
        BufferedImage buffImg = ProcessingUtils.PImageToBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);
        try (ZooModel<Image, DetectedObjects> model = this.criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                // detect objects
                DetectedObjects objects = predictor.predict(img);
                // parse DetectedObjects to a list of MLObject
                MLObject[] objectList = DetectedObjectsToMLObjects(objects, originalImgWidth, originalImgHeight);
                return objectList;
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
     * Run object detection on given PImage and save output image
     * @param pImg, saveOutputImg, fileName
     * @return MLObject[]
     */
    public MLObject[] detect(PImage pImg, Boolean saveOutputImg, String fileName) {
        // get original image size
        int orgImgW = pImg.width;
        int orgImgH = pImg.height;
        // convert PImage to DJL Image
        BufferedImage buffImg = ProcessingUtils.PImageToBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);
        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                // detect objects
                DetectedObjects detected = predictor.predict(img);
                // parse DetectedObjects to a list of MLObject
                MLObject[] results = DetectedObjectsToMLObjects(detected, orgImgW, orgImgH);
                // save bounding box image
                if (saveOutputImg == true) {
                    DJLUtils.saveBoundingBoxImage(this.parent, fileName, img, detected);
                }
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

//    public void saveBoundingBoxImage(PImage pImg, MLObject[] objectList, String fileName) {
//        // get original image size
//        int originalImgWidth = pImg.width;
//        int originalImgHeight = pImg.height;
//        // convert PImage to DJL Image
//        BufferedImage buffImg = ProcessingUtils.PImageToBuffImage(pImg);
//        Image img = ImageFactory.getInstance().fromImage(buffImg);
//
//        DetectedObjects objects = DetectedObjectsToMLObjects(objectList, originalImgWidth, originalImgHeight);
//        DJLUtils.saveBoundingBoxImage(this.parent, fileName, img, objects);
//    }
}
