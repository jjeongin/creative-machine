package ml;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ml.util.ProcessingUtils;
import ml.MLPose;
import ml.MLKeyPoint;

public class PoseDetector {
    PApplet parent; // reference to the parent sketch
    private Predictor<Image, Joints> predictor;
    private Predictor<Image, DetectedObjects> objectDetectorPredictor;
    private static final Logger logger =
            LoggerFactory.getLogger(ml.PoseDetector.class);
    public PoseDetector(PApplet myParent) {
        this.parent = myParent;
        logger.info("model loading..");
        // load pose detector model
        // initialize criteria to load the model
        Criteria<Image, Joints> criteria = Criteria.builder()
                    .optApplication(Application.CV.POSE_ESTIMATION)
                    .setTypes(Image.class, Joints.class)
                    .optFilter("backbone", "resnet18")
                    .optFilter("flavor", "v1b")
                    .optFilter("dataset", "imagenet")
                    .optEngine("MXNet")
                    .build();
        // load the model
        ZooModel<Image, Joints> model = null;
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
        // load object detector model
        // set a criteria to load an object detection model
        Criteria<Image, DetectedObjects> objectDetectorCriteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("backbone", "mobilenet1.0")
                        .optFilter("dataset", "voc")
                        .optEngine("MXNet")
                        .build();
        // load the model
        ZooModel<Image, DetectedObjects> objectDetectorModel = null;
        try {
            objectDetectorModel = objectDetectorCriteria.loadModel();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ModelNotFoundException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }
        // initialize a predictor for the model
        this.objectDetectorPredictor = objectDetectorModel.newPredictor();
        logger.info("successfully loaded!");
    }

    private MLPose JointsToMLPose(Joints joints, float personTopLeftX, float personTopLeftY, float personWidth, float personHeight) {
        List<Joints.Joint> jointList = joints.getJoints();
        int numJoints = jointList.size();
        List<MLKeyPoint> keyPoints = new ArrayList<MLKeyPoint>();
        for (int i = 0; i < numJoints; i++) {
            // retrieve information from a key point
            float x = (float) jointList.get(i).getX();
            x = x * personWidth + personTopLeftX;
            float y = (float) jointList.get(i).getY();
            y = y * personHeight + personTopLeftY;
            float confidence = (float) jointList.get(i).getConfidence(); // get probability
            // add each object to the list as keypoint
            MLKeyPoint keyPoint = new MLKeyPoint(x, y, confidence);
            keyPoints.add(keyPoint);
        }
        MLPose pose = new MLPose(keyPoints);
        return pose;
    }

    public MLPose predict(PImage pImg) {
        BufferedImage buffImg = ProcessingUtils.PImageToBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);
        // find a person in image using object detection
        int width = img.getWidth();
        int height = img.getHeight();
        Rectangle personRect = predictPersonInImage(img);
        // throw error if no person is found
        if (personRect == null) {
//            logger.warn("No person found in image.");
            return new MLPose(Collections.emptyList());
        }
        // convert cropped rectangle to image
        float personTopLeftX = (float) personRect.getX() * width;
        float personTopLeftY = (float) personRect.getY() * height;
        float personWidth = (float) (personRect.getWidth() * width);
        float personHeight = (float) (personRect.getHeight() * height);
        Image personImg = img.getSubImage(
                        (int) (Math.max(personTopLeftX, 0)),
                        (int) (Math.max(personTopLeftY, 0)),
                        (int) (Math.min(personWidth, width - personTopLeftX)),
                        (int) (Math.min(personHeight, height - personTopLeftY)));
        // detect pose on the person image
        Joints joints = predictJointsInPerson(personImg);
        MLPose pose = JointsToMLPose(joints, personTopLeftX, personTopLeftY, personWidth, personHeight);
        return pose;
    }

    private Rectangle predictPersonInImage(Image img) {
        // detect objects in image
        DetectedObjects detectedObjects = null;
        try {
            detectedObjects = this.objectDetectorPredictor.predict(img);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        // crop out a person
        List<DetectedObjects.DetectedObject> items = detectedObjects.items();
        for (DetectedObjects.DetectedObject item : items) {
            if ("person".equals(item.getClassName())) {
                Rectangle rect = item.getBoundingBox().getBounds();
                return rect;
            }
        }
        return null;
    }

    private Joints predictJointsInPerson(Image person) {
        // find joints from the person image
        Joints joints = null;
        try {
            joints = this.predictor.predict(person);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        return joints;
    }
}
