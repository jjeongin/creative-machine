import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import java.util.List;

DeepVision deepVision = new DeepVision(this);
YOLONetwork yolo;
List<ObjectDetectionResult> detections;

PImage image;

public void setup() {
  size(640, 480, FX2D);

  colorMode(HSB, 360, 100, 100);
  
  image = loadImage("hk.jpg");

  println("creating model...");
  yolo = deepVision.createYOLOv3Tiny();

  println("loading yolo model...");
  yolo.setup();
}

public void draw() {
  background(55);

  image(image, 0, 0);

  yolo.setConfidenceThreshold(0.2f);
  detections = yolo.run(image);

  noFill();
  strokeWeight(2f);

  for (ObjectDetectionResult detection : detections) {
    stroke((int)(360.0 / yolo.getNames().size() * detection.getClassId()), 80, 100);
    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());
  }

  surface.setTitle("YOLO Test - FPS: " + Math.round(frameRate));
}