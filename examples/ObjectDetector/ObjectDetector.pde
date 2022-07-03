import ml.*;

ObjectDetectorDJL detectorDJL;
String outputName;
PImage orgImg;
DetectedObjectDJL[] detectedList;

void setup() {
    size(768, 576);

    String modelName = "cocossd"; // "openimages_ssd", "cocossd", or "yolo"
    String imgName = "dog_bike_car"; // "dog_bike_car" or "kite_people"

    detectorDJL = new ObjectDetectorDJL(this, modelName);
    orgImg = loadImage( "data/" + imgName + ".jpeg");

    // run object detection and save output image
    outputName = imgName + "_output_" + modelName + ".png";
    detectedList = detectorDJL.detect(orgImg, true, outputName);

    // print each detected object and its probability
    for (int i = 0; i < detectedList.length; i++) {
        println(detectedList[i].getName() + " detected! (probability: " + detectedList[i].getProbability() + ")");
    }
}

void draw() {
   // draw bounding boxes of detected objects on the original image
   image(orgImg, 0, 0);
   noFill();
   stroke(255, 0, 0);
   for (int i = 0; i < detectedList.length; i++) {
       DetectedObjectDJL obj = detectedList[i];
       // multiply by image width & height to draw each bounding box on right position
       rect(obj.getUpperLeft().x * orgImg.width, obj.getUpperLeft().y * orgImg.height,
               obj.getWidth() * orgImg.width, obj.getHeight() * orgImg.height);
   }
}