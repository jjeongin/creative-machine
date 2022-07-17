package ml;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;

import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import processing.core.PApplet;

import ml.translator.FaceDetectorTranslator;
import processing.core.PImage;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class FaceDetector {
    PApplet parent; // reference to the parent sketch

    private Criteria<Image, DetectedObjects> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(FaceDetector.class);

    public FaceDetector(PApplet myParent, String modelNameOrURL) {
        this.parent = myParent;
        logger.info("model loading..");

        double confThresh = 0.85f;
        double nmsThresh = 0.45f;
        double[] variance = {0.1f, 0.2f};
        int topK = 5000;
        int[][] scales = {{16, 32}, {64, 128}, {256, 512}};
        int[] steps = {8, 16, 32};

        FaceDetectorTranslator translator =
                new FaceDetectorTranslator(confThresh, nmsThresh, variance, topK, scales, steps);

        this.criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optModelUrls(modelNameOrURL)
                .optModelName("R50") // prefix of .params and .json files inside model dir or archive
                .optTranslator(translator)
                .optEngine("MXNet") // Use MXNet engine
                .build();

        logger.info("successfully loaded!");
    }

    public DetectedObjects detect(PImage pImg) {
        Path facePath = Paths.get("data/largest_selfie.jpeg");
        Image img = null;
        try {
            img = ImageFactory.getInstance().fromFile(facePath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                saveBoundingBoxImage(img, detection);
                return detection;
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

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        img.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("ultranet_detected.png");
        img.save(Files.newOutputStream(imagePath), "png");
        logger.info("Face detection result image has been saved in: {}", imagePath);
    }
}
