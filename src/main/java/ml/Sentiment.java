package ml;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import processing.core.PApplet;

import java.io.IOException;

import ml.translator.SentimentTranslator;

public class Sentiment {
    PApplet parent; // reference to the parent sketch

    private Criteria<String, Classifications> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(Sentiment.class);

    public Sentiment(PApplet myParent) {
        this.parent = myParent;
        logger.info("model loading..");

        // load distilbert model finetuned on SST2
        String modelURL = "file:///Users/jlee/src/ml4processing/models/distilbert-sst2/saved_model/";

        this.criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.SENTIMENT_ANALYSIS)
                        .setTypes(String.class, Classifications.class)
                        .optModelUrls(modelURL)
                        .optTranslator(new SentimentTranslator())
                        .optEngine("TensorFlow")
                        // This model was traced on CPU and can only run on CPU
                        .optDevice(Device.cpu())
                        .build();

        logger.info("successfully loaded!");
    }

    public Classifications predict(String input) {
        try (ZooModel<String, Classifications> model = this.criteria.loadModel()) {
            Predictor<String, Classifications> predictor = model.newPredictor();
            return predictor.predict(input);
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

