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

import ml.translator.SentimentTranslator;
import ml.MLObject;
import ml.util.ProcessingUtils;

public class Sentiment {
    PApplet parent; // reference to the parent sketch

    private Criteria<String, Classifications> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(Sentiment.class);

    public Sentiment(PApplet myParent, String modelNameOrURL) {
        this.parent = myParent;
        logger.info("model loading..");

        if (modelNameOrURL.equals("distilbert")) {
            modelNameOrURL = "https://www.dropbox.com/s/j8hkvqqm4a9awcy/distilbert-sst2.zip?dl=1";
        }

        this.criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.SENTIMENT_ANALYSIS)
                        .setTypes(String.class, Classifications.class)
                        .optModelUrls(modelNameOrURL)
                        .optModelName(ProcessingUtils.getFileNameFromPath(modelNameOrURL)+"/saved_model")
                        .optTranslator(new SentimentTranslator())
                        .optEngine("TensorFlow")
                        .build();

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
        try (ZooModel<String, Classifications> model = this.criteria.loadModel()) {
            Predictor<String, Classifications> predictor = model.newPredictor();
            // run sentiment analysis
            Classifications classified = predictor.predict(input);
            // parse Classifications to a list of MLObject
            MLObject[] results = ClassificationsToMLObjects(classified);
            return results;
        } catch (ModelNotFoundException e) {
            throw new RuntimeException(e);
        } catch (MalformedModelException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }
}

