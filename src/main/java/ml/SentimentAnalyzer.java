package ml;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import processing.core.PApplet;

import java.io.IOException;
import java.util.List;

import ml.translator.SentimentAnalyzerTranslator;
import ml.util.ProcessingUtils;
import ml.MLLabel;

public class SentimentAnalyzer {
    PApplet parent; // reference to the parent sketch
    private Predictor<String, Classifications> predictor;
    private static final Logger logger =
            LoggerFactory.getLogger(ml.SentimentAnalyzer.class);

    public SentimentAnalyzer(PApplet myParent) {
        this.parent = myParent;
        logger.info("model loading..");
        // DistilBERT base uncased finetuned SST-2: sentiment analysis model that returns negative and positive score of a input sentence
        // original model from https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
        String modelURL =  "https://www.dropbox.com/s/j8hkvqqm4a9awcy/distilbert-sst2.zip?dl=1";
        // load a custom model with URL
//        // check if the URL is valid (source: https://stackoverflow.com/questions/2230676/how-to-check-for-a-valid-url-in-java)
//        URL url = null; // check for the URL protocol
//        try {
//            url = new URL(modelNameOrURL);
//        } catch (MalformedURLException e) {
//            throw new RuntimeException(e);
//        }
//        try {
//            url.toURI(); // extra check if the URI is valid
//        } catch (URISyntaxException e) {
//            throw new RuntimeException(e);
//        }
//        modelNameOrURL = String.valueOf(url);
        // initialize criteria to load the model
        Criteria<String, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.SENTIMENT_ANALYSIS)
                        .setTypes(String.class, Classifications.class)
                        .optModelUrls(modelURL)
                        .optModelName(ProcessingUtils.getFileNameFromPath(modelURL)+"/saved_model")
                        .optTranslator(new SentimentAnalyzerTranslator())
                        .optEngine("TensorFlow")
                        .build();
        // load the model
        ZooModel<String, Classifications> model = null;
        try {
            model = criteria.loadModel();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ModelNotFoundException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        }
        // initialize a predictor for the model
        this.predictor = model.newPredictor();
        logger.info("successfully loaded!");
    }

    private MLLabel[] ClassificationsToMLLabels(Classifications classified) {
        List<Classifications.Classification> classifications = classified.items();
        MLLabel[] labels = new MLLabel[2]; // [Negative, Positive]
        for (int i = 0; i < 2; i++) {
            // retrieve information from a classified object
            String labelName = classifications.get(i).getClassName(); // get class name
            float confidence = (float) classifications.get(i).getProbability(); // get probability
            // add each object to the list as MLObject
            labels[i] = new MLLabel(labelName, confidence);
        }
        return labels;
    }

    public MLLabel[] predict(String input) {
        // run sentiment analysis
        Classifications classified = null;
        try {
            classified = this.predictor.predict(input);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        // parse Classifications to a list of MLObject
        MLLabel[] results = ClassificationsToMLLabels(classified);
        return results;
    }
}

