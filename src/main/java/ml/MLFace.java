package ml;

import java.util.List;
import processing.core.PVector;

import ml.MLObject;
import ml.MLKeyPoint;

public class MLFace extends MLObject {
    List<MLKeyPoint> keyPoints;

    public MLFace(String label, float confidence, float x, float y, float width, float height, List<MLKeyPoint> keyPoints) {
        super(label, confidence, x, y, width, height);
        this.keyPoints = keyPoints;
    }

    public MLFace(float confidence, float x, float y, float width, float height, List<MLKeyPoint> keyPoints) {
        super("Face", confidence, x, y, width, height);
        this.keyPoints = keyPoints;
    }

    public List<MLKeyPoint> getKeyPoints() {
        return this.keyPoints;
    }

    public MLKeyPoint getLeftEye() {
        return this.keyPoints.get(0);
    }

    public MLKeyPoint getRightEye() {
        return this.keyPoints.get(1);
    }

    public MLKeyPoint getNose() {
        return this.keyPoints.get(2);
    }

    public MLKeyPoint getLeftMouth() {
        return this.keyPoints.get(3);
    }

    public MLKeyPoint getRightMouth() {
        return this.keyPoints.get(4);
    }
}
