// Use this package name when debugging from IntelliJ
package main.java.ml.model.ObjectDetector;

// Use this package name when building with gradle to release the library
//package ml.model.ObjectDetector;

import ai.djl.*;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import processing.core.PApplet;
import processing.core.PImage;

import static processing.core.PConstants.RGB;

/**
 * Object Detector using Deep Java Library
 * @example ObjectDetectorDJLExample
 *
 */
public class ObjectDetectorDJL {
    PApplet parent; // reference to the parent sketch
    Criteria<Image, DetectedObjects> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(ObjectDetectorDJL.class);

    /**
     * constructor
     * @param myParent
     * @param modelNameOrURL : model name to load (choose from object detection models in tf model zoo)
     *                         if not in the model zoo, try to load as URL
     */
    public ObjectDetectorDJL(PApplet myParent, String modelNameOrURL) {
        parent = myParent;
        logger.info("model loading..");

//        String modelUrl =
//                "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz";

//        String backbone = "mobilenet_v2"; // default
//        if (modelNameOrURL.equals("mobilenet_v2")) {
//            backbone =  modelNameOrURL;
//        }

        String backbone;
        if ("TensorFlow".equals(Engine.getDefaultEngineName())) {
            backbone = "mobilenet_v2";
        } else {
            backbone = "resnet50";
        }

        criteria = Criteria.builder()
                .optApplication(Application.CV.OBJECT_DETECTION)
                .setTypes(Image.class, DetectedObjects.class)
                .optFilter("backbone", backbone)
                .optEngine(Engine.getDefaultEngineName())
                .build();

        logger.info("successfully loaded!");
    }

    /**
     * Convert PImage to BufferedImage (from Processing source code)
     * @param pImg
     * @return BufferedImage
     */
    public BufferedImage PImagetoBuffImage(PImage pImg) {
        pImg.loadPixels();
        int type = (pImg.format == RGB) ?
                BufferedImage.TYPE_INT_RGB : BufferedImage.TYPE_INT_ARGB;
        BufferedImage image =
                new BufferedImage(pImg.pixelWidth, pImg.pixelHeight, type);
        WritableRaster wr = image.getRaster();
        wr.setDataElements(0, 0, pImg.pixelWidth, pImg.pixelHeight, pImg.pixels);
        return image;
    }

    /**
     * Run object detection on given PImage
     * @param pImg
     * @return DetectedObjects
     */
    public String detect(PImage pImg) {
        BufferedImage buffImg = PImagetoBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detected = predictor.predict(img);
                return detected.toString();
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

    public String detect(PImage pImg, Boolean saveOutputImg, String fileName) {
        BufferedImage buffImg = PImagetoBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detected = predictor.predict(img);
                if (saveOutputImg == true) {
                    saveBoundingBoxImage(fileName, img, detected);
                }
                return detected.toString();
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
