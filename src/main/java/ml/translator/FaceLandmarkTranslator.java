package ml.translator;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Point;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;

import ai.djl.translate.TranslatorContext;
import ml.MLKeyPoint;

import java.util.ArrayList;
import java.util.List;

public class FaceLandmarkTranslator implements Translator<Image, MLKeyPoint[]> {
    private int width;
    private int height;
    public FaceLandmarkTranslator() {

    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
        this.width = input.getWidth();
        this.height = input.getHeight();

        array = array.transpose(2, 0, 1).flip(0); // H, W, C RGB -> C, H, W RGB -> C, H, W BGR

        // The network by default takes float32
        if (!array.getDataType().equals(DataType.FLOAT32)) {
            array = array.toType(DataType.FLOAT32, false);
        }
        NDArray mean =
                ctx.getNDManager().create(new float[] {104f, 117f, 123f}, new Shape(3, 1, 1));
        array = array.sub(mean); // shape : (channel, image_h, image_w)
        return new NDList(array);
    }

    @Override
    public MLKeyPoint[] processOutput(TranslatorContext ctx, NDList list){
        NDArray landms = list.get(0);
        landms.reshape(-1,2); // shape: (272, )
        System.out.println(landms);

//        List<Point> keyPoints = new ArrayList<>();
        MLKeyPoint[] landmarkPoints = new MLKeyPoint[68];
        for (int i = 0; i < 68; i++) { // x, y for 68 landmarks
            float x = landms.getFloat(i * 2);
            float y = landms.getFloat(i * 2 + 1);
            landmarkPoints[i] = new MLKeyPoint(x * width, y * height);
        }
        return landmarkPoints;
    }
}
