import ml.*;
import processing.video.*;

Capture cam;
ObjectDetector detector;

void setup() {
  size(600, 400);
  // load an object detector
  // * Model Options
  // - SSD Models: "openimages_ssd", "coco_ssd", "voc_ssd"
  // - YOLO Models: "coco_yolo", "voc_yolo"
  String modelName = "coco_ssd"; // choose the model you want to use
  detector = new ObjectDetector(this, modelName);
  // load and start the camera
  cam = new Capture(this, "pipeline:autovideosrc");
  cam.start();
}

void draw() {
  // read from the camera
  if (cam.available()) {
    cam.read(); // read new frame
  }
  // display the captured image
  image(cam, 0, 0);
  if (cam.width == 0) {
    return;
  }
  // detect objects in the captured image
  MLObject[] output = detector.predict(cam);
  // draw a bounding box of each detected object
  noFill();
  stroke(255, 0, 0);
  for (int i = 0; i < output.length; i++) {
      MLObject obj = output[i];
      rect(obj.getX(), obj.getY(), obj.getWidth(), obj.getHeight());
  }
}