import ml.*;

SentimentAnalyzer analyzer;
MLObject[] prediction;

void setup() {
    size(450, 250);
    background(255);
    PFont font = createFont("Arial", 20);
    textFont(font);

    // load model
    analyzer = new SentimentAnalyzer(this);

    // define input
    String input = "Machine Learning is fun.";

    // run sentiment analysis
    prediction = analyzer.predict(input);
}

void draw() {
    // print Negative score (0 to 1)
    fill(0, 102, 153); // dark blue
    text(prediction[0].getLabel() + " Score: " + prediction[0].getConfidence(), 40, 70);

    // print Positive score (0 to 1)
    fill(213, 240, 10); // light green
    text(prediction[1].getLabel() + " Score: " + prediction[1].getConfidence(), 40, 130);

    // print sentiment with higher score
    fill(121, 129, 132); // grey
    // if negative score is bigger than positive score
    if (prediction[0].getConfidence() > prediction[1].getConfidence()) {
        text("Sentiment is Negative ; (", 40, 200);
    }
    else {
        text("Sentiment is Positive : >", 40, 200);
    }
}
