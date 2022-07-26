package ml.translator;

import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.Dictionary;
import java.util.List;
import java.util.function.ToIntFunction;
import java.util.stream.LongStream;

public class SentimentTranslator implements Translator<String, Classifications> {
    private Vocabulary vocabulary;
    private BertTokenizer tokenizer;

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
            Model model = ctx.getModel();
            URL url = model.getArtifact("vocab.txt"); // load vocab list from local
//        URL url = new URL("https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/vocab.txt");
        vocabulary =
                DefaultVocabulary.builder().addFromTextFile(url).optUnknownToken("[UNK]").build();
        tokenizer = new BertTokenizer();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        List<String> tokens = tokenizer.tokenize(input);
        int[] indices = tokens.stream().mapToInt(t -> (int) vocabulary.getIndex(t)).toArray();
        int[] attentionMask = new int[tokens.size()];
        Arrays.fill(attentionMask, 1);

        NDManager manager = ctx.getNDManager();
        NDArray attentionMaskArray = manager.create(attentionMask);
        NDArray indicesArray = manager.create(indices);
        return new NDList(attentionMaskArray, indicesArray);
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray raw = list.singletonOrThrow();
        NDArray computed = raw.exp().div(raw.exp().sum(new int[] {0}, true)); // apply softmax
        return new Classifications(Arrays.asList("Negative", "Positive"), computed);
    }
}
