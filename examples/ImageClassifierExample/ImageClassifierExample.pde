import ml.*;

ImageClassifier classifier;
PImage img;
MLLabel[] results;

void setup() {
    size(400, 450);

    // load model
    String modelName = "Darknet"; // "MobileNet", "Darknet"
    classifier = new ImageClassifier(this, modelName);

    // load image
    img = loadImage("bird.jpeg");

    // classify
    results = classifier.predict(img);

    // print all the labels (by default, results contain top 5 labels with the highest confidence)
    for (int i = 0; i < results.length; i++) {
        println("Label " + String.valueOf(i+1) + ": " + results[i].getLabel() + ", Confidence: " + results[i].getConfidence());
    }
}

void draw() {
    background(255);
    image(img, 0, 0);
    fill(0);
    textSize(15);
    text("Top Label: " + results[0].getLabel() + "\nConfidence: " + results[0].getConfidence(), 10, 420);
}
