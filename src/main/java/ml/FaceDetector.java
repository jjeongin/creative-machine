package ml;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.*;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;

import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import processing.core.PApplet;
import processing.core.PImage;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import ml.MLFace;
import ml.MLKeyPoint;
import ml.translator.FaceDetectorTranslator;
import ml.translator.FaceLandmarkTranslator;
import ml.util.DJLUtils;
import ml.util.ProcessingUtils;

import processing.core.PVector;

import static processing.core.PConstants.RGB;

public class FaceDetector {
    PApplet parent; // reference to the parent sketch
    private Predictor<Image, DetectedObjects> predictor; // default 5 landmark model
    private Predictor<Image, MLKeyPoint[]> landmarkPredictor; // 68 landmark model
    private boolean detectLandmarksOrNot; // if true, detect 68 landmarks

    private static final Logger logger =
            LoggerFactory.getLogger(FaceDetector.class);
    /**
    Default Constructor
     */
    public FaceDetector(PApplet myParent) {
        this.parent = myParent;
        logger.info("model loading..");
        // default config for the translator
        double confThresh = 0.85f;
        double nmsThresh = 0.45f;
        double[] variance = {0.1f, 0.2f};
        int topK = 5000;
        int[][] scales = {{16, 32}, {64, 128}, {256, 512}};
        int[] steps = {8, 16, 32}; // strides
        // initialize the translator
        FaceDetectorTranslator translator =
                new FaceDetectorTranslator(confThresh, nmsThresh, variance, topK, scales, steps);
        // default 5 landmarks model
        // face detection model that returns bounding box of the face and 5 facial landmarks
        String modelURL = "https://www.dropbox.com/s/b9rvq4cz4tniw5e/retinaface-mobilenet.zip?dl=1";
        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optModelUrls(modelURL)
                .optModelName(ProcessingUtils.getFileNameFromPath(modelURL)+"/saved_model")
                .optTranslator(translator)
                .optEngine("TensorFlow") // Use TensorFlow engine
                .build();
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

//    /**
//     * Constructor with an optional argument to detect 68 face landmarks
//     * @param detectLandmarksOrNot Detect 68 landmarks if true
//     */
//    public FaceDetector(PApplet myParent, Boolean detectLandmarksOrNot) {
//        this.parent = myParent;
//        logger.info("model loading..");
//        // store the boolean value
//        this.detectLandmarksOrNot = detectLandmarksOrNot;
//        // default config for the translator
//        double confThresh = 0.85f;
//        double nmsThresh = 0.45f;
//        double[] variance = {0.1f, 0.2f};
//        int topK = 5000;
//        int[][] scales = {{16, 32}, {64, 128}, {256, 512}};
//        int[] steps = {8, 16, 32}; // strides
//        // initialize the translator
//        FaceDetectorTranslator translator =
//                new FaceDetectorTranslator(confThresh, nmsThresh, variance, topK, scales, steps);
//        // face detection model that returns bounding box of the face and 5 facial landmarks
//        String modelURL = "https://www.dropbox.com/s/b9rvq4cz4tniw5e/retinaface-mobilenet.zip?dl=1";
//        // initialize criteria to load the model
//        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
//                .setTypes(Image.class, DetectedObjects.class)
//                .optModelUrls(modelURL)
//                .optModelName(ProcessingUtils.getFileNameFromPath(modelURL)+"/saved_model")
//                .optTranslator(translator)
//                .optEngine("TensorFlow") // Use TensorFlow engine
//                .build();
//        // load the model
//        ZooModel<Image, DetectedObjects> model = null;
//        try {
//            model = criteria.loadModel();
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        } catch (ModelNotFoundException e) {
//            throw new RuntimeException(e);
//        } catch (MalformedModelException e) {
//            throw new RuntimeException(e);
//        }
//        // initialize a predictor for the model
//        this.predictor = model.newPredictor();
//        // WIP ----------------------------------------------
//        // load landmark detection model that identifies 68 landmarks
//        if (this.detectLandmarksOrNot == true) { // 68 landmarks
//            String landmarkModelNameOrURL = "https://www.dropbox.com/s/fwfwwlrl7uqk6ey/retinaface-mobilenet-68landmarks.zip?dl=1";
//            FaceLandmarkTranslator landmarkTranslator = new FaceLandmarkTranslator();
//            Criteria<Image, MLKeyPoint[]> landmarkCriteria = Criteria.builder()
//                    .setTypes(Image.class, MLKeyPoint[].class)
//                    .optModelUrls(landmarkModelNameOrURL)
//                    .optModelName(ProcessingUtils.getFileNameFromPath(landmarkModelNameOrURL)+"/saved_model")
//                    .optTranslator(landmarkTranslator)
//                    .optEngine("TensorFlow") // Use TensorFlow engine
//                    .build();
//            ZooModel<Image, MLKeyPoint[]> landmarkModel = null;
//            try {
//                landmarkModel = landmarkCriteria.loadModel();
//            } catch (IOException e) {
//                throw new RuntimeException(e);
//            } catch (ModelNotFoundException e) {
//                throw new RuntimeException(e);
//            } catch (MalformedModelException e) {
//                throw new RuntimeException(e);
//            }
//            DJLUtils.printInputOutputInfo(landmarkModel); // TEST
//            this.landmarkPredictor = landmarkModel.newPredictor();
//        }
//        // ----------------------------------------------
//        logger.info("successfully loaded!");
//    }

    private MLFace[] DetectedObjectsToMLFaces(DetectedObjects detected, int orgImgW, int orgImgH) {
        int biggerDimension = Math.max(orgImgW, orgImgH);
        int numObjects = detected.getNumberOfObjects();
        MLFace[] faces = new MLFace[numObjects]; // initialize a new array
        for (int i = 0; i < numObjects; i++) {
            // get the ith detected object
            DetectedObjects.DetectedObject d = detected.item(i);
            // retrieve information from a detected object
            String className = d.getClassName(); // get class name
            float probability = (float) d.getProbability(); // get probability
            Rectangle bound = d.getBoundingBox().getBounds(); // get bounding box
            float x = (float) bound.getX() * biggerDimension;
            float y = (float) bound.getY() * biggerDimension;
            //  PVector upperLeft = new PVector((float) bound.getX(), (float) bound.getY()); // get upper left corner of the bounding box
            float width = (float) bound.getWidth() * biggerDimension; // get width of the bounding box
            float height = (float) bound.getHeight() * biggerDimension; // get height of the bounding box
            // convert landmark points
            List<MLKeyPoint> landmarks = new ArrayList<>(); // create new landmark list
            Iterable<Point> landmarkPoints = d.getBoundingBox().getPath(); // get landmark points from Landmark object
            for (Point p : landmarkPoints) {
                MLKeyPoint l = new MLKeyPoint((float) p.getX()/640*biggerDimension, (float) p.getY()/640*biggerDimension);
                landmarks.add(l); // add each element in the array
            }
            // add each object to the list as MLFace
            faces[i] = new MLFace(className, probability, x, y, width, height, landmarks);
        }
        return faces;
    }

    private PImage resizeToSquare(PImage orgImg) {
        // copy resized image to a new square image
        PImage squareImg = this.parent.createImage(640, 640, RGB);
        // resize bigger side to 640 (default input size is 640 x 640 for RetinaFace model)
        if (orgImg.width >= orgImg.height) {
            squareImg.copy(orgImg, 0, 0, orgImg.width, orgImg.height, 0, 0, 640, 640*orgImg.height/orgImg.width);
        } else {
            squareImg.copy(orgImg, 0, 0, orgImg.width, orgImg.height, 0, 0, 640*orgImg.width/orgImg.height, 640);
        }
        return squareImg;
    }

//    private MLFace[] detect68Landmarks(Image img, DetectedObjects detected, int orgImgW, int orgImgH) {
//        int biggerDimension = Math.max(orgImgW, orgImgH);
//        // retrieve cropped face images from resized img using bounding boxes in DetectedObjects
//        int numObjects = detected.getNumberOfObjects();
//        MLFace[] faces = new MLFace[numObjects];
//        for (int i = 0; i < numObjects; i++) {
//            // get the ith detected object
//            DetectedObjects.DetectedObject d = detected.item(i);
//            // retrieve information from a detected object
//            String className = d.getClassName(); // get class name
//            float probability = (float) d.getProbability(); // get probability
//            Rectangle bound = d.getBoundingBox().getBounds(); // get bounding box
//            float x = (float) bound.getX() * biggerDimension;
//            float y = (float) bound.getY() * biggerDimension;
//            //  PVector upperLeft = new PVector((float) bound.getX(), (float) bound.getY()); // get upper left corner of the bounding box
//            float width = (float) bound.getWidth() * biggerDimension; // get width of the bounding box
//            float height = (float) bound.getHeight() * biggerDimension; // get height of the bounding box
//            // crop a face from img using d's bounding box
//            System.out.println(img.getWidth() + " | " + img.getHeight());
////                    System.out.println(bound.getX()*640 + " " + (int) bound.getY()*640 + " " + (int) bound.getWidth()*640 + " " + (int) bound.getHeight()*640);
//            Image croppedFace = img.getSubImage((int) (bound.getX()*640), (int) (bound.getY()*640), (int) (bound.getWidth()*640), (int) (bound.getHeight()*640));
//            // convert landmark points
//            List<MLKeyPoint> landmarks = new ArrayList<>(); // create new landmark list
//            MLKeyPoint[] landmarkPoints = new MLKeyPoint[0];
//            try {
//                landmarkPoints = this.landmarkPredictor.predict(croppedFace);
//            } catch (TranslateException e) {
//                throw new RuntimeException(e);
//            }
//            for (MLKeyPoint p : landmarkPoints) {
//                MLKeyPoint l = new MLKeyPoint((float) p.getX()/640*biggerDimension, (float) p.getY()/640*biggerDimension);
//                landmarks.add(l); // add each element in the array
//            }
//            // add each object to the list as MLFace
//            faces[i] = new MLFace(className, probability, x, y, width, height, landmarks);
//        }
//        return faces;
//    }

    public MLFace[] detect(PImage pImg) {
        // resize original image to square
        PImage squareImg = resizeToSquare(pImg);
        // convert PImage to Image
        BufferedImage buffImg = ProcessingUtils.PImageToBuffImage(squareImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);
        // detect faces with confidence, bounding box, and 5 landmarks
        DetectedObjects detected = null;
        try {
            detected = this.predictor.predict(img);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        // convert DJL DetectedObjects to MLFace
        MLFace[] detectedList = DetectedObjectsToMLFaces(detected, pImg.width, pImg.height);;
//        if (detected != null) {
//            // detect extra 68 landmarks if detectLandmarksOrNot is true
//            if (this.detectLandmarksOrNot == true) {
//            // detectedList = detect68Landmarks(img, detected, pImg.width, pImg.height);
//            }
//            else { // if only 5 landmarks
//                // parse DetectedObjects to a list of MLFace
//                detectedList = DetectedObjectsToMLFaces(detected, pImg.width, pImg.height);
//            }
//        }
        return detectedList;
    }
}
