package ml;

import processing.core.PVector;

public class MLObject {
    private String label;
    private float confidence, width, height;
    private PVector upperLeft;

    public MLObject(String label, float confidence, PVector upperLeft, float width, float height) {
        this.label = label;
        this.confidence = confidence;
        // TO DO : change these to x, y, width, height in original image scale like ml5
        //         change these to public variable ?
        this.upperLeft = upperLeft; // value scaled from 0 to 1
        this.width = width; // value scaled from 0 to 1
        this.height = height; // value scaled from 0 to 1
    }

    public MLObject(String label, float confidence) {
        this.label = label;
        this.confidence = confidence;
    }

    public String getLabel() {
        return this.label;
    }

    public float getConfidence() {
        return this.confidence;
    }

    public PVector getUpperLeft() {
        return this.upperLeft;
    }

    public float getWidth() {
        return this.width;
    }

    public float getHeight() {
        return this.height;
    }
}
