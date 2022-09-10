import ml.*;
import processing.video.*;

Capture cam;
PoseDetector poseDetector;

public void setup() {
  size(600, 400);
  // load the model
  poseDetector = new PoseDetector(this);
  // load and start the camera
  cam = new Capture(this);
  cam.start();
}

public void draw() {
  // read from the camera
  if (cam.available()) {
    cam.read(); // read new frame
  }
  // display captured image
  image(cam, 0, 0);
  if (cam.width == 0) {
    return;
  }

  // detect a pose
  MLPose pose = poseDetector.predict(cam);
  // plot each keypoint from the detected pose
  for (int i = 0; i < pose.getKeyPoints().size(); i++) {
      MLKeyPoint keyPoint = pose.getKeyPoints().get(i);
      stroke(255, 0, 0);
      strokeWeight(10);
      point(keyPoint.getX(), keyPoint.getY());
  }
}