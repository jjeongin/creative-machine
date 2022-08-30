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
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.List;

import ml.translator.SentimentTranslator;
import ml.MLObject;
import ml.util.ProcessingUtils;

public class Sentiment {
    PApplet parent; // reference to the parent sketch
    private Predictor<String, Classifications> predictor;
    private static final Logger logger =
            LoggerFactory.getLogger(Sentiment.class);

    public Sentiment(PApplet myParent, String modelNameOrURL) {
        this.parent = myParent;
        logger.info("model loading..");

        if (modelNameOrURL.equals("distilbert")) {
            modelNameOrURL = "https://www.dropbox.com/s/j8hkvqqm4a9awcy/distilbert-sst2.zip?dl=1";
        }
        else { // load model with remote url
            // check if the URL is valid (source: https://stackoverflow.com/questions/2230676/how-to-check-for-a-valid-url-in-java)
            URL url = null; // check for the URL protocol
            try {
                url = new URL(modelNameOrURL);
            } catch (MalformedURLException e) {
                throw new RuntimeException(e);
            }
            try {
                url.toURI(); // extra check if the URI is valid
            } catch (URISyntaxException e) {
                throw new RuntimeException(e);
            }
            modelNameOrURL = String.valueOf(url);
        }

        Criteria<String, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.SENTIMENT_ANALYSIS)
                        .setTypes(String.class, Classifications.class)
                        .optModelUrls(modelNameOrURL)
                        .optModelName(ProcessingUtils.getFileNameFromPath(modelNameOrURL)+"/saved_model")
                        .optTranslator(new SentimentTranslator())
                        .optEngine("TensorFlow")
                        .build();

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
        this.predictor = model.newPredictor();

        logger.info("successfully loaded!");
    }

    private MLObject[] ClassificationsToMLObjects(Classifications classified) {
        List<Classifications.Classification> classifications = classified.items();
        MLObject[] objectList = new MLObject[2]; // [Negative, Positive]
        for (int i = 0; i < 2; i++) {
            // retrieve information from a classified object
            String labelName = classifications.get(i).getClassName(); // get class name
            float confidence = (float) classifications.get(i).getProbability(); // get probability
            // add each object to the list as MLObject
            objectList[i] = new MLObject(labelName, confidence);
        }
        return objectList;
    }

    public MLObject[] predict(String input) {
        // run sentiment analysis
        Classifications classified = null;
        try {
            classified = this.predictor.predict(input);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        // parse Classifications to a list of MLObject
        MLObject[] results = ClassificationsToMLObjects(classified);
        return results;
    }
}

