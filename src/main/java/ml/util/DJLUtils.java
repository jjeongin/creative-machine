package ml.util;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.util.PairList;
import processing.core.PApplet;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DJLUtils {
    /**
     * Save Bounding Box image with DetectedObjects
     * @param fileName
     * @param img
     * @param detected
     */
    public static void saveBoundingBoxImage(PApplet parent, String fileName, Image img, DetectedObjects detected) {
        // set output path as the Processing sketch's directory
        Path outputDir = Paths.get(parent.sketchPath());
        try {
            Files.createDirectories(outputDir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // draw bounding boxes with DJL function
        img.drawBoundingBoxes(detected);

        // set filename
        Path imagePath;
        if (fileName != null) { // use user-specified filename
            imagePath = outputDir.resolve(fileName);
        }
        else { // if filename is null, use default filename
            imagePath = outputDir.resolve("output.png");
        }

        // OpenJDK can't save jpg with alpha channel
        try {
            img.save(Files.newOutputStream(imagePath), "png");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Output image has been saved in: " + imagePath);
    }

    public static <T, K> void printInputOutputInfo(ZooModel<T, K> model) {
        // print model input & output info
        PairList<String, Shape> inputInfo = model.describeInput();
        System.out.println("input info");
        System.out.println(inputInfo.keys());
        System.out.println(inputInfo.values());

        PairList<String, Shape> outputInfo = model.describeOutput();
        System.out.println("output info");
        System.out.println(outputInfo.keys());
        System.out.println(outputInfo.values());
    }
}
