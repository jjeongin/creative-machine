import ml.model.*;

ObjectDetectorDJL detectorDJL;
String outputName;

void setup() {
    size(768, 576);
    detectorDJL = new ObjectDetectorDJL(this, "mobilenet_v2");
    String imgPath = "data/";
    PImage img = loadImage(imgPath + "dog_bike_car.jpeg");

    // run object detection
    DetectedObjects detected = detectorDJL.detect(img);

    // draw bounding boxes on original image
    outputName = "dog_bike_car_output.png";
    detectorDJL.saveBoundingBoxImage(imgPath, outputName, img, detected);

    println("Detected objects: " + detected);
}

void draw() {
    // display output image
    PImage outputImg = loadImage(outputName);
    image(outputImg, 0, 0);
}
