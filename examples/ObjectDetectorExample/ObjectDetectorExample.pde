import ml.*;

ObjectDetector detector;
PImage img;
MLObject[] output;

void setup() {
    size(768, 576);
    // * Model Options
    // - SSD Models: "openimages_ssd", "coco_ssd", "voc_ssd"
    // - YOLO Models: "coco_yolo", "voc_yolo"
    // choose the model you want to use
    String modelName = "coco_ssd";
    // load an object detector
    detector = new ObjectDetector(this, modelName);
    // load an input image
    String imgName = "dog_bike_car"; // "dog_bike_car" or "kite_people"
    img = loadImage(imgName + ".jpeg");
    // run the object detection and save the output image
    output = detector.predict(img, "data/" + imgName + "_output_" + modelName + ".png");
    // print a label and confidence score of each object
    for (int i = 0; i < output.length; i++) {
        println(output[i].getLabel() + " detected! (confidence: " + output[i].getConfidence() + ")");
    }
}

void draw() {
   // draw a bounding box of each object
   image(img, 0, 0);
   noFill();
   stroke(255, 0, 0);
   for (int i = 0; i < output.length; i++) {
       MLObject obj = output[i];
       rect(obj.getX(), obj.getY(), obj.getWidth(), obj.getHeight());
   }
}