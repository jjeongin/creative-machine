package ml.util;

import processing.core.PImage;

import java.io.File;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;

import static processing.core.PConstants.RGB;

public class ProcessingUtils {
    /**
     * Convert PImage to BufferedImage (from Processing source code)
     * @param pImg
     * @return BufferedImage
     */
    public static BufferedImage PImageToBuffImage(PImage pImg) {
        pImg.loadPixels();
        int type = (pImg.format == RGB) ?
                BufferedImage.TYPE_INT_RGB : BufferedImage.TYPE_INT_ARGB;
        BufferedImage image =
                new BufferedImage(pImg.pixelWidth, pImg.pixelHeight, type);
        WritableRaster wr = image.getRaster();
        wr.setDataElements(0, 0, pImg.pixelWidth, pImg.pixelHeight, pImg.pixels);
        return image;
    }

    public static String getFileNameFromPath(String path) {
        File file = new File(path);
        String name = file.getName();
        if (name == null || name.isEmpty()) {
            return "unnamed";
        }

        // From https://www.baeldung.com/java-filename-without-extension
        String extPattern = "(?<!^)[.].*";
        return name.replaceAll(extPattern, "");
    }
}
