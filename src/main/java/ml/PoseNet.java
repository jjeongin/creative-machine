package ml;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
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

public class PoseNet {
    PApplet parent; // reference to the parent sketch
    private Criteria<Image, Joints> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(PoseNet.class);

    public PoseNet(PApplet myParent, String modelNameOrURL) {
        this.parent = myParent;
        logger.info("model loading..");

        if (modelNameOrURL.equals("ResNet")) {
            this.criteria = Criteria.builder()
                    .optApplication(Application.CV.POSE_ESTIMATION)
                    .setTypes(Image.class, Joints.class)
                    .optFilter("backbone", "resnet18")
                    .optFilter("flavor", "v1b")
                    .optFilter("dataset", "imagenet")
                    .optEngine("MXNet")
                    .build();
        }

        logger.info("successfully loaded!");
    }

    private MLPose parseJoints(Joints joints, float personTopLeftX, float personTopLeftY, float personWidth, float personHeight) {
        List<Joints.Joint> jointList = joints.getJoints();
        int numJoints = jointList.size();
        List<MLKeyPoint> keypoints = new ArrayList<MLKeyPoint>();

        for (int i = 0; i < numJoints; i++) {
            // retrieve information from a key point
            float x = (float) jointList.get(i).getX();
            x = x * personWidth + personTopLeftX;
            float y = (float) jointList.get(i).getY();
            y = y * personHeight + personTopLeftY;
            float confidence = (float) jointList.get(i).getConfidence(); // get probability

            // add each object to the list as keypoint
            MLKeyPoint keypoint = new MLKeyPoint(x, y, confidence);
            keypoints.add(keypoint);
        }

        MLPose pose = new MLPose(keypoints);
        return pose;
    }

    public MLPose predict(PImage pImg) {
        BufferedImage buffImg = ProcessingUtils.PImagetoBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);

        // find a person in image using object detection
        int width = img.getWidth();
        int height = img.getHeight();
        Rectangle personRect = predictPersonInImage(img);

        // convert cropped rectangle to image
        float personTopLeftX = (float) personRect.getX() * width;
        float personTopLeftY = (float) personRect.getY() * height;
        float personWidth = (float) (personRect.getWidth() * width);
        float personHeight = (float) (personRect.getHeight() * height);

        Image personImg = img.getSubImage(
                        (int) (personTopLeftX),
                        (int) (personTopLeftY),
                        (int) (personWidth),
                        (int) (personHeight));

        // throw error if person is not found
        if (personImg == null) {
            logger.warn("No person found in image.");
            return new MLPose(Collections.emptyList());
        }

        // detect pose on the person image
        Joints joints = predictJointsInPerson(personImg);
        MLPose pose = parseJoints(joints, personTopLeftX, personTopLeftY, personWidth, personHeight);
        return pose;
    }

    private Rectangle predictPersonInImage(Image img) {
        // load object detection model
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("size", "512")
                        .optFilter("backbone", "resnet50")
                        .optFilter("flavor", "v1")
                        .optFilter("dataset", "voc")
                        .optEngine("MXNet")
                        .build();

        // detect objects
        DetectedObjects detectedObjects;
        try (ZooModel<Image, DetectedObjects> ssd = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = ssd.newPredictor()) {
                detectedObjects = predictor.predict(img);
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
        try (ZooModel<Image, Joints> pose = this.criteria.loadModel()) {
            try (Predictor<Image, Joints> predictor = pose.newPredictor()) {
                Joints joints = predictor.predict(person);
//                saveJointsImage(person, joints);
                return joints;
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

    private void saveJointsImage(Image img, Joints joints) {
        // Default output path is parent sketch directory
        Path outputDir = Paths.get(this.parent.sketchPath());
        try {
            Files.createDirectories(outputDir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        img.drawJoints(joints); // draw joints on the person image

        String fileName = "joints.png";
        Path imagePath = outputDir.resolve(fileName);
        // OpenJDK can't save jpg with alpha channel
        try {
            img.save(Files.newOutputStream(imagePath), "png");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        logger.info("Detected pose image has been saved in: {}", imagePath);
    }

}
