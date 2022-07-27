import ml.*;

Sentiment sentiment;
MLObject[] prediction;

void setup() {
    size(450, 250);
    background(255);
    PFont font = createFont("Arial", 20);
    textFont(font);

    // load model
    sentiment = new Sentiment(this);

    // define input
    String input = "Machine Learning is fun.";

    // run sentiment analysis
    prediction = sentiment.predict(input);
}

void draw() {
    // print Negative score
    fill(23, 144, 209); // blue
    text(prediction[0].getLabel() + " Score: " + prediction[0].getConfidence(), 40, 70);

    // print Positive score
    fill(255, 153, 235); // pink
    text(prediction[1].getLabel() + " Score: " + prediction[1].getConfidence(), 40, 130);

    fill(0);
    if (prediction[0].getConfidence() > prediction[1].getConfidence()) {
        text("Sentiment is Negative : (", 40, 200);
    }
    else {
        text("Sentiment is Positive : )", 40, 200);
    }
}
