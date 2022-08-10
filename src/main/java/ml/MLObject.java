package ml;

import processing.core.PVector;

public class MLObject {
    private String label;
    private float confidence, width, height, x, y;
//    private PVector upperLeft;

    public MLObject(String label, float confidence, float x, float y, float width, float height) {
        this.label = label;
        this.confidence = confidence; // confidence score of the detection (form 0 to 1)
        this.x = x; // x coordinate of the upper left corner of the bounding box
        this.y = y; // y coordinate of the upper left corner of the bounding box
        this.width = width; // width of the bounding box
        this.height = height; // height of the bounding box
    }

    public MLObject(String label, float confidence) {
        this.label = label;
        this.confidence = confidence; // confidence score of the classification (form 0 to 1)
    }

    public String getLabel() {
        return this.label;
    }

    public float getConfidence() {
        return this.confidence;
    }

    public float getX() {
        return this.x;
    }

    public float getY() {
        return this.y;
    }

    public float getWidth() {
        return this.width;
    }

    public float getHeight() {
        return this.height;
    }
}
