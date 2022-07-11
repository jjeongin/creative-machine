import ml.*;

PoseNet poseNet;
PImage img;
MLPose pose;

public void setup() {
    size(1102, 932);
    background(255);

    // load model
    poseNet = new PoseNet(this, "ResNet");

    // load image
    img = loadImage("data/pose_soccer.png");

    // detect pose
    pose = poseNet.predict(img);
}

public void draw() {
    // display output image
    // PImage output = loadImage("joints.png");
    // image(output, 0, 0);

    // display original image
    image(img, 0, 0);

    // plot each keypoint from the detected pose
    for (int i = 0; i < pose.getKeyPoints().size(); i++) {
        MLKeyPoint keypoint = pose.getKeyPoints().get(i);

        stroke(255, 0, 0);
        strokeWeight(10);
        point(keypoint.getX(), keypoint.getY());
    }
}
