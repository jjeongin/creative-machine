package ml;

import java.util.List;
import processing.core.PVector;

import ml.MLObject;
import ml.MLKeyPoint;

public class MLFace extends MLObject {
    List<MLKeyPoint> landmarks;

    public MLFace(String label, float confidence, float x, float y, float width, float height, List<MLKeyPoint> landmarks) {
        super(label, confidence, x, y, width, height);
        this.landmarks = landmarks;
    }

    public MLFace(float confidence, float x, float y, float width, float height, List<MLKeyPoint> landmarks) {
        super("Face", confidence, x, y, width, height);
        this.landmarks = landmarks;
    }

    public List<MLKeyPoint> getLandmarks() {
        return this.landmarks;
    }
}
