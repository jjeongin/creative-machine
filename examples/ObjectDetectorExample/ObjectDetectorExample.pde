import ml.*;

ObjectDetector detector;
PImage img;
MLObject[] output;
String outputName;

void setup() {
    size(768, 576);

    // load object detector
    String modelName = "cocossd"; // "openimages_ssd", "cocossd", or "yolo"
    detector = new ObjectDetector(this, modelName);

    // load image
    String imgName = "dog_bike_car"; // "dog_bike_car" or "kite_people"
    img = loadImage(imgName + ".jpeg");

    // run object detection and save output image
    outputName = "data/" + imgName + "_output_" + modelName + ".png";
    output = detector.detect(img, true, outputName);

    // print label and confidence of each object
    for (int i = 0; i < output.length; i++) {
        println(output[i].getLabel() + " detected! (confidence: " + output[i].getConfidence() + ")");
    }
}

void draw() {
   // draw bounding box of each object
   image(img, 0, 0);
   noFill();
   stroke(255, 0, 0);
   for (int i = 0; i < output.length; i++) {
       MLObject obj = output[i];
       rect(obj.getX(), obj.getY(), obj.getWidth(), obj.getHeight());
   }
}