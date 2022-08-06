import ml.*;

FaceDetector detector;
PImage img;
MLFace[] faces;

void setup() {
    size(1600, 898);
    background(255);

    // load model
    detector = new FaceDetector(this);

    // load image
    img = loadImage("data/largest_selfie.jpeg");

    // detect faces
    faces = detector.detect(img);
}

void draw() {
    // draw faces
    image(img, 0, 0);
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
            PVector landmark = face.getLandmarks().get(j);
            circle(landmark.x, landmark.y, 5);
        }
    }
}