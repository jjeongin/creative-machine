import ml.*;

ObjectDetectorDJL detectorDJL;
String outputName;
PImage orgImg;
MLObject[] results;

void setup() {
    size(768, 576);

    String modelName = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz";
    String imgName = "dog_bike_car"; // "dog_bike_car" or "kite_people"

    detectorDJL = new ObjectDetectorDJL(this, modelName);
    orgImg = loadImage( "data/" + imgName + ".jpeg");

    // run object detection and save output image
    outputName = imgName + "_output_from_url.png";
    results = detectorDJL.detect(orgImg, true, outputName);

    // print each detected object and its probability
    for (int i = 0; i < results.length; i++) {
        println(results[i].getLabel() + " detected! (probability: " + results[i].getConfidence() + ")");
    }
}

void draw() {
   // draw bounding boxes of detected objects on the original image
   image(orgImg, 0, 0);
   noFill();
   stroke(255, 0, 0);
   for (int i = 0; i < results.length; i++) {
       MLObject obj = results[i];
       // multiply by image width & height to draw each bounding box on right position
       rect(obj.getUpperLeft().x * orgImg.width, obj.getUpperLeft().y * orgImg.height,
               obj.getWidth() * orgImg.width, obj.getHeight() * orgImg.height);
   }
}