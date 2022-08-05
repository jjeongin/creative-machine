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
import ai.djl.training.util.DownloadUtils;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import processing.core.PApplet;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import ml.translator.SentimentTranslator;
import ml.MLObject;

public class Sentiment {
    PApplet parent; // reference to the parent sketch

    private Criteria<String, Classifications> criteria; // model

    private static final Logger logger =
            LoggerFactory.getLogger(Sentiment.class);

    public Sentiment(PApplet myParent) {
        this.parent = myParent;
        logger.info("model loading..");

        // load distilbert model finetuned on SST2
        String modelNameorURL = "distilbert-sst2/saved_model/";

        String sketchPath = myParent.sketchPath();
        String modelPath = sketchPath + "/" + modelNameorURL;
        logger.info("model found at: " + modelPath);

        // another method?
//        Path modelPath = Paths.get(modelNameorURL);
//        Path absolutePath = modelPath.toAbsolutePath();
//        System.out.println(absolutePath);

        this.criteria =
                Criteria.builder()
                        .optApplication(Application.NLP.SENTIMENT_ANALYSIS)
                        .setTypes(String.class, Classifications.class)
                        .optModelUrls(modelPath)
                        .optTranslator(new SentimentTranslator())
                        .optEngine("TensorFlow")
//                        // This model was traced on CPU and can only run on CPU
//                        .optDevice(Device.cpu())
                        .build();

        logger.info("successfully loaded!");
    }

    private MLObject[] parseClassifications(Classifications classified) {
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
            MLObject[] results = parseClassifications(classified);
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

