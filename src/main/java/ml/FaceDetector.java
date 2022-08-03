package ml;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
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

import ml.translator.FaceDetectorTranslator;
import ml.util.ProcessingUtils;

public class FaceDetector {
    PApplet parent; // reference to the parent sketch

    private Criteria<Image, DetectedObjects> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(FaceDetector.class);

    public FaceDetector(PApplet myParent, String modelNameOrURL) {
        this.parent = myParent;
        logger.info("model loading..");

        double confThresh = 0.85f;
//        double confThresh = 0.5f;
        double nmsThresh = 0.45f;
        double[] variance = {0.1f, 0.2f};
        int topK = 5000;
        int[][] scales = {{16, 32}, {64, 128}, {256, 512}};
        int[] steps = {8, 16, 32}; // strides

        FaceDetectorTranslator translator =
                new FaceDetectorTranslator(confThresh, nmsThresh, variance, topK, scales, steps);

        this.criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optModelUrls(modelNameOrURL)
//                .optModelName("R50") // prefix of .params and .json files inside model dir or archive
                .optTranslator(translator)
                .optEngine("TensorFlow") // Use MXNet engine
                .build();

        logger.info("successfully loaded!");
    }

    public DetectedObjects detect(PImage pImg) {
        BufferedImage buffImg = ProcessingUtils.PImagetoBuffImage(pImg);
        Image img = ImageFactory.getInstance().fromImage(buffImg);

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                PairList<String, Shape> inputInfo = model.describeInput();
                System.out.println("input info");
                System.out.println(inputInfo.keys());
                System.out.println(inputInfo.values());

                PairList<String, Shape> outputInfo = model.describeOutput();
                System.out.println("output info");
                System.out.println(outputInfo.keys());
                System.out.println(outputInfo.values());

                DetectedObjects detection = predictor.predict(img);
//                DetectedObjects detection = predictor.predict(img);
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

    private void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get(this.parent.sketchPath());
        Files.createDirectories(outputDir);

        img.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("retinaface_detected.png");
        img.save(Files.newOutputStream(imagePath), "png");
        logger.info("Face detection result image has been saved in: {}", imagePath);
    }
}
