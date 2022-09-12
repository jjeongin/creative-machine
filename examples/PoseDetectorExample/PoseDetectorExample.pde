import ml.*;

PoseDetector poseDetector;
PImage img;
MLPose pose;

public void setup() {
    size(1102, 932);

    // load model
    poseDetector = new PoseDetector(this);

    // load image
    img = loadImage("pose_soccer.png");

    // detect pose
    pose = poseDetector.predict(img);
}

public void draw() {
    background(255);
    // display original image
    image(img, 0, 0);
    // plot each keypoint from the detected pose
    for (int i = 0; i < pose.getKeyPoints().size(); i++) {
        MLKeyPoint keyPoint = pose.getKeyPoints().get(i);
        stroke(255, 0, 0);
        strokeWeight(10);
        point(keyPoint.getX(), keyPoint.getY());
    }
}
