import ml.*;
import processing.video.*;

Capture cam;
FaceDetector detector;

void setup() {
  size(600, 400);
  // load face detector
  detector = new FaceDetector(this);
  // load and start camera
  cam = new Capture(this);
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
    stroke(243, 176, 255);
    rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());
    // draw landmarks
    noStroke();
    fill(255, 131, 110);
    for (int j = 0; j < face.getLandmarks().size(); j++) {
        MLKeyPoint landmark = face.getLandmarks().get(j);
        circle(landmark.getX(), landmark.getY(), 5);
    }
  }
}