package ml;

import ml.MLKeyPoint;
import java.util.List;

public class MLPose {
    private List<MLKeyPoint> keypoints;

    public MLPose(List<MLKeyPoint> keypoints) {
        this.keypoints = keypoints;
    }

    public List<MLKeyPoint> getKeyPoints() {
        return this.keypoints;
    }

    public MLKeyPoint getNose() {
        return this.keypoints.get(0);
    }

    public MLKeyPoint getLeftEye() {
        return this.keypoints.get(1);
    }

    public MLKeyPoint getRightEye() {
        return this.keypoints.get(2);
    }

    public MLKeyPoint getLeftEar() {
        return this.keypoints.get(3);
    }

    public MLKeyPoint getRightEar() {
        return this.keypoints.get(4);
    }

    public MLKeyPoint getLeftShoulder() {
        return this.keypoints.get(5);
    }

    public MLKeyPoint getRightShoulder() {
        return this.keypoints.get(6);
    }

    public MLKeyPoint getLeftElbow() {
        return this.keypoints.get(7);
    }

    public MLKeyPoint getRightElbow() {
        return this.keypoints.get(8);
    }

    public MLKeyPoint getLeftWrist() {
        return this.keypoints.get(9);
    }

    public MLKeyPoint getRightWrist() {
        return this.keypoints.get(10);
    }

    public MLKeyPoint getLeftHip() {
        return this.keypoints.get(11);
    }

    public MLKeyPoint getRightHip() {
        return this.keypoints.get(12);
    }

    public MLKeyPoint getLeftKnee() {
        return this.keypoints.get(13);
    }

    public MLKeyPoint getRightKnee() {
        return this.keypoints.get(14);
    }

    public MLKeyPoint getLeftAnkle() {
        return this.keypoints.get(15);
    }

    public MLKeyPoint getRightAnkle() {
        return this.keypoints.get(16);
    }
}
