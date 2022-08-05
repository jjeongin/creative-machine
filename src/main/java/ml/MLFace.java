package ml;

import java.util.List;
import processing.core.PVector;

import ml.MLObject;

public class MLFace extends MLObject {
    List<PVector> landmarks;

    public MLFace(String label, float confidence, float x, float y, float width, float height, List<PVector> landmarks) {
        super(label, confidence, x, y, width, height);
        this.landmarks = landmarks;
    }

    public MLFace(float confidence, float x, float y, float width, float height, List<PVector> landmarks) {
        super("Face", confidence, x, y, width, height);
        this.landmarks = landmarks;
    }

    public List<PVector> getLandmarks() {
        return this.landmarks;
    }
}
