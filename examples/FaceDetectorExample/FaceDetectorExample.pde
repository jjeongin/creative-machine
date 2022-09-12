import ml.*;

FaceDetector detector;
PImage img;
MLFace[] faces;

void setup() {
    size(800, 533);
    background(255);

    // load face detector model
    detector = new FaceDetector(this);

    // load image
    img = loadImage("friends.jpg");

    // detect faces
    faces = detector.predict(img);
}

void draw() {
    // draw faces
    image(img, 0, 0);
    for (int i = 0; i < faces.length; i++) {
        // get each face
        MLFace face = faces[i];
        // draw bounding box
        noFill();
        stroke(240, 121, 81);
        rect(face.getX(), face.getY(), face.getWidth(), face.getHeight());
        // draw each facial landmark
        noStroke();
        fill(250, 255, 112);
        for (int j = 0; j < face.getKeyPoints().size(); j++) {
            MLKeyPoint keyPoint = face.getKeyPoints().get(j);
            circle(keyPoint.getX(), keyPoint.getY(), 5);
        }
    }
}