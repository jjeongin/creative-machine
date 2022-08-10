import ml.*;

ObjectDetector detector;
PImage img;
MLObject[] output;
String outputName;

void setup() {
    size(768, 576);

    // load object detector from remote url
    String modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz";
    detector = new ObjectDetector(this, modelURL);

    // load image
    String imgName = "dog_bike_car"; // "dog_bike_car" or "kite_people"
    img = loadImage("data/" + imgName + ".jpeg");

    // run object detection and save output image
    outputName = imgName + "_output_from_url.png";
    results = detector.detect(img, true, outputName);

    // print label and confidence of each object
    for (int i = 0; i < results.length; i++) {
        println(results[i].getLabel() + " detected! (confidence: " + results[i].getConfidence() + ")");
    }
}

void draw() {
   // draw bounding box of each object
   image(img, 0, 0);
   noFill();
   stroke(255, 0, 0);
   for (int i = 0; i < results.length; i++) {
       MLObject obj = results[i];
       rect(obj.getX(), obj.getY(), obj.getWidth(), obj.getHeight());
   }
}