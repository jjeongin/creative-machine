# Sentiment
Analyze sentiment (Negative and Positive) of the provided sentence.

<img src="../data/sentiment_demo.png" width="500">

## Quick Start
```
// create a Sentiment analysis model
Sentiment sentiment = new new Sentiment(this, "distilbert");

// define input sentence
String input = "Machine Learning is fun.";

// run sentiment analysis
MLObject[] output = sentiment.predict(input);
```

## Usage
### Initialize
```
Sentiment sentiment = new new Sentiment(this, modelNameOrURL);
```
#### Parameters
String modelNameOrURL: (required) Can be a model name of built-in models ("distilbert") or a remote url/file path to a parent directory containing TensorFlow saved_model folder
### Methods
predict(String input): Runs sentiment analysis on input String and returns an array of [MLObject]() with two sentiment labels (Negative and Positive) and confidence scores.
```
String input = "Machine Learning is fun.";

// analyze sentiment
MLObject[] output = sentiment.predict(input);

// print Negative score (0 to 1)
println("Sentiment: " + output[0].getLabel() + ", Confidence: " + output[0].getConfidence());

// print Positive score (0 to 1)
println("Sentiment: " + output[1].getLabel() + ", Confidence: " + output[1].getConfidence());
```
***Input***
- String input: (required) String to analyze the sentiment.

***Output***
- MLObject[]: List of [MLObject](). Contains 2 labels (Negative and Positive) with confidence scores (from 0 to 1). 

## Examples
[SentimentExample](https://github.com/jjeongin/ml4processing/tree/master/examples/SentimentExample)
