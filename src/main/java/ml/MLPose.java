package ml;

import ml.MLKeyPoint;
import java.util.List;

public class MLPose {
    private List<MLKeyPoint> keyPoints;

    public MLPose(List<MLKeyPoint> keyPoints) {
        this.keyPoints = keyPoints;
    }

    public List<MLKeyPoint> getKeyPoints() {
        return this.keyPoints;
    }

    public MLKeyPoint getNose() {
        return this.keyPoints.get(0);
    }

    public MLKeyPoint getLeftEye() {
        return this.keyPoints.get(1);
    }

    public MLKeyPoint getRightEye() {
        return this.keyPoints.get(2);
    }

    public MLKeyPoint getLeftEar() {
        return this.keyPoints.get(3);
    }

    public MLKeyPoint getRightEar() {
        return this.keyPoints.get(4);
    }

    public MLKeyPoint getLeftShoulder() {
        return this.keyPoints.get(5);
    }

    public MLKeyPoint getRightShoulder() {
        return this.keyPoints.get(6);
    }

    public MLKeyPoint getLeftElbow() {
        return this.keyPoints.get(7);
    }

    public MLKeyPoint getRightElbow() {
        return this.keyPoints.get(8);
    }

    public MLKeyPoint getLeftWrist() {
        return this.keyPoints.get(9);
    }

    public MLKeyPoint getRightWrist() {
        return this.keyPoints.get(10);
    }

    public MLKeyPoint getLeftHip() {
        return this.keyPoints.get(11);
    }

    public MLKeyPoint getRightHip() {
        return this.keyPoints.get(12);
    }

    public MLKeyPoint getLeftKnee() {
        return this.keyPoints.get(13);
    }

    public MLKeyPoint getRightKnee() {
        return this.keyPoints.get(14);
    }

    public MLKeyPoint getLeftAnkle() {
        return this.keyPoints.get(15);
    }

    public MLKeyPoint getRightAnkle() {
        return this.keyPoints.get(16);
    }
}
