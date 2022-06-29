// Use this package name when debugging from IntelliJ
package main.java.ml.model.ObjectDetector;

// Use this package name when building with gradle to release the library
//package ml;

import processing.core.PVector;

public class DetectedObjectDJL {
    private String name;
    private float probability, width, height;
    private PVector upperLeft;

    public DetectedObjectDJL(String name, float probability, PVector upperLeft, float width, float height) {
        this.name = name;
        this.probability = probability;
        this.upperLeft = upperLeft;
        this.width = width;
        this.height = height;
    }

    public String getName() {
        return this.name;
    }

    public float getProbability() {
        return this.probability;
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
