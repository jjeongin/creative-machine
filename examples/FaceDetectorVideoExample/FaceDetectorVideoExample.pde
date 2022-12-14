import ml.*;
import processing.video.*;

Capture cam;
FaceDetector detector;

void setup() {
  size(600, 400);
  // load face detector
  detector = new FaceDetector(this);
  // load and start camera
  cam = new Capture(this, width, height);
  cam.start();
}

void draw() {
  // read from the camera
  if (cam.available()) {
    cam.read(); // read new frame
  }
  // display captured image
  image(cam, 0, 0);
  if (cam.width == 0) {
    return;
  }

  // detect faces
  MLFace[] faces = detector.predict(cam);
  // draw bounding boxes of detected faces
  for (int i = 0; i < faces.length; i++) {
    // get each face
    MLFace face = faces[i];
    // draw bounding box
    noFill();
    stroke(240, 121, 81);
    rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());
    // draw draw each facial landmark
    noStroke();
    fill(250, 255, 112);
    for (int j = 0; j < face.getKeyPoints().size(); j++) {
        MLKeyPoint keyPoint = face.getKeyPoints().get(j);
        circle(keyPoint.getX(), keyPoint.getY(), 5);
    }
  }
}