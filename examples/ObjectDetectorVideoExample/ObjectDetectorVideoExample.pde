import ml.*;
import processing.video.*;

Capture cam;
ObjectDetector detector;

void setup() {
  size(600, 400);

  // load object detector
  String modelName = "cocossd"; // "openimages_ssd", "cocossd", or "yolo"
  detector = new ObjectDetector(this, modelName);

  // load and start camera
  cam = new Capture(this, "pipeline:autovideosrc");
  cam.start();
}

void draw() {
  if (cam.available()) {
    cam.read(); // read new frame
  }

  // display captured image
  image(cam, 0, 0);
  if (cam.width == 0) {
    return;
  }

  // draw bounding boxes of detected objects
  MLObject[] output = detector.detectPreloaded(cam);
  noFill();
  stroke(255, 0, 0);
  for (int i = 0; i < output.length; i++) {
      MLObject obj = output[i];
      rect(obj.getX(), obj.getY(), obj.getWidth(), obj.getHeight());
  }
}