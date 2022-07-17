package ml.translator;

import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

public class SentimentTranslator implements Translator<String, Classifications> {
    private Vocabulary vocabulary;
    private BertTokenizer tokenizer;
//    private Sigmoid(outputs) {
//        return 1.0 / (1.0 + np.exp(-_outputs))
//    }

//
//    def softmax(_outputs):
//    maxes = np.max(_outputs, axis=-1, keepdims=True)
//    shifted_exp = np.exp(_outputs - maxes)
//            return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
//            Model model = ctx.getModel();
//            URL url = model.getArtifact("distilbert-base-uncased-finetuned-sst-2-english-vocab.txt");
        URL url = new URL("https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/vocab.txt");
        vocabulary =
                DefaultVocabulary.builder().addFromTextFile(url).optUnknownToken("[UNK]").build();
        tokenizer = new BertTokenizer();
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray raw = list.singletonOrThrow();
        System.out.println("raw");
        System.out.println(new Classifications(Arrays.asList("Negative", "Positive"), raw));

        NDArray sigmoid = raw.neg().exp().add(1.0).pow(-1); // sigmoid
        System.out.println("sigmoid");
        System.out.println(new Classifications(Arrays.asList("Negative", "Positive"), sigmoid));

        NDArray computed = raw.exp().div(raw.exp().sum(new int[] {0}, true)); // softmax
        System.out.println("softmax");

        return new Classifications(Arrays.asList("Negative", "Positive"), computed);
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        List<String> tokens = tokenizer.tokenize(input);
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
        long[] attentionMask = new long[tokens.size()];
        Arrays.fill(attentionMask, 1);
        NDManager manager = ctx.getNDManager();
        NDArray indicesArray = manager.create(indices);
        NDArray attentionMaskArray = manager.create(attentionMask);
        return new NDList(indicesArray, attentionMaskArray);
    }
}
