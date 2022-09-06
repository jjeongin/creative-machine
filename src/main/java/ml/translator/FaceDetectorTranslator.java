/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package ml.translator;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.*;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static ai.djl.modality.cv.util.NDImageUtils.resize;

public class FaceDetectorTranslator implements Translator<Image, DetectedObjects> {

    private double confThresh;
    private double nmsThresh;
    private int topK;
    private double[] variance;
    private int[][] scales;
    private int[] steps;
    private int width;
    private int height;
    private boolean landmarks68;


    public FaceDetectorTranslator(
            double confThresh,
            double nmsThresh,
            double[] variance,
            int topK,
            int[][] scales,
            int[] steps,
            boolean landmarks68) {
        this.confThresh = confThresh;
        this.nmsThresh = nmsThresh;
        this.variance = variance;
        this.topK = topK;
        this.scales = scales;
        this.steps = steps;
        this.landmarks68 = landmarks68;
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
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        //  NDList list has 9 elements (3 different strides, each stride outputs its scores, bboxes, landmarks)
        NDManager manager = ctx.getNDManager();
        double scaleXY = variance[0];
        double scaleWH = variance[1];

        // get probabilities
        NDArray prob = list.get(1).get(":, 1:"); // shape: (16800, 1)
        prob =                                          // shape: (2, 16800)
                NDArrays.stack(
                        new NDList(
                                prob.argMax(1).toType(DataType.FLOAT32, false), // index
                                prob.max(new int[]{1})));                            // value

        // get face bounding boxes
        NDArray boxRecover = boxRecover(manager, width, height, scales, steps);
        NDArray boundingBoxes = list.get(2);
        NDArray bbWH = boundingBoxes.get(":, 2:").mul(scaleWH).exp().mul(boxRecover.get(":, 2:"));
        NDArray bbXY =
                boundingBoxes
                        .get(":, :2")
                        .mul(scaleXY)
                        .mul(boxRecover.get(":, 2:"))
                        .add(boxRecover.get(":, :2"))
                        .sub(bbWH.mul(0.5f));

        boundingBoxes = NDArrays.concat(new NDList(bbXY, bbWH), 1);

        // get landmarks
        NDArray landms = list.get(0);
        landms = decodeLandm(landms, boxRecover, scaleXY); // decode 5 landmarks

        // filter the result with the threshold
        NDArray cutOff = prob.get(1).gt(confThresh);
        boundingBoxes = boundingBoxes.transpose().booleanMask(cutOff, 1).transpose();
        landms = landms.transpose().booleanMask(cutOff, 1).transpose();
        prob = prob.booleanMask(cutOff, 1);

        // start categorical filtering
        long[] order = prob.get(1).argSort().get(":" + topK).toLongArray();
        prob = prob.transpose();
        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();
        Map<Integer, List<BoundingBox>> recorder = new ConcurrentHashMap<>();

        for (int i = order.length - 1; i >= 0; i--) {
            long currMaxLoc = order[i];
            float[] classProb = prob.get(currMaxLoc).toFloatArray();
            int classId = (int) classProb[0];
            double probability = classProb[1];
            // changed type from double to float
            float[] boxArr = boundingBoxes.get(currMaxLoc).toFloatArray();
            float[] landmsArr = landms.get(currMaxLoc).toFloatArray();
            Rectangle rect = new Rectangle(boxArr[0], boxArr[1], boxArr[2], boxArr[3]);
            List<BoundingBox> boxes = recorder.getOrDefault(classId, new ArrayList<>());
            boolean belowIoU = true;
            for (BoundingBox box : boxes) {
                if (box.getIoU(rect) > nmsThresh) {
                    belowIoU = false;
                    break;
                }
            }
            if (belowIoU) {
                List<Point> keyPoints = new ArrayList<>(); // list of landmarks

                // Only add 5 landmarks if the landmarks68 is false
                // if landmarks68 is true, an empty list of keyPoints will be passed
                if (!this.landmarks68) {
                    // add each landmark coordinate into keyPoints
                    for (int j = 0; j < landms.size(1) / 2; j++) { // 5 face landmarks
                        float x = landmsArr[j * 2];
                        float y = landmsArr[j * 2 + 1];
                        keyPoints.add(new Point(x * width, y * height));
                    }
                }

                // four points of a bounding box and the list of all the landmarks
                Landmark landmark =
                        new Landmark(boxArr[0], boxArr[1], boxArr[2], boxArr[3], keyPoints);

                boxes.add(landmark);
                recorder.put(classId, boxes);
                String className = "Face";
                retNames.add(className);
                retProbs.add(probability);
                retBB.add(landmark);
            }
        }

        return new DetectedObjects(retNames, retProbs, retBB);
    }

    private NDArray boxRecover(
            NDManager manager, int width, int height, int[][] scales, int[] steps) {
        int[][] aspectRatio = new int[steps.length][2]; // shape : (3, 2)
        for (int i = 0; i < steps.length; i++) { // add aspected ratios for each step (=stride)
            int wRatio = (int) Math.ceil( (float) width / steps[i]);
            int hRatio = (int) Math.ceil( (float) height / steps[i]);
            aspectRatio[i] = new int[] {hRatio, wRatio};
        }

        // Changed double -> float list for defaultBoxes
        // since using double causes a type-mismatch error when TFNDArray runs clip(0, 1);
        List<float[]> defaultBoxes = new ArrayList<>();
        for (int idx = 0; idx < steps.length; idx++) {
            int[] scale = scales[idx]; // for each scale, {{16, 32}, {64, 128}, {256, 512}}
            for (int h = 0; h < aspectRatio[idx][0]; h++) {
                for (int w = 0; w < aspectRatio[idx][1]; w++) {
                    for (int i : scale) {
                        float skx = (float) (i * 1.0 / width);
                        float sky = (float) (i * 1.0 / height);
                        float cx = (float) ((w + 0.5) * steps[idx] / width);
                        float cy = (float) ((h + 0.5) * steps[idx] / height);
                        defaultBoxes.add(new float[] {cx, cy, skx, sky});
                    }
                }
            }
        }

        float[][] boxes = new float[defaultBoxes.size()][defaultBoxes.get(0).length];
        for (int i = 0; i < defaultBoxes.size(); i++) {
            boxes[i] = defaultBoxes.get(i);
        }
        return manager.create(boxes).clip(0.0, 1.0); // return boxes in scale from 0 to 1
    }

    // decode face landmarks, 5 points per face
    private NDArray decodeLandm(NDArray pre, NDArray priors, double scaleXY) {
        NDArray point1 =
                pre.get(":, :2").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        NDArray point2 =
                pre.get(":, 2:4").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        NDArray point3 =
                pre.get(":, 4:6").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        NDArray point4 =
                pre.get(":, 6:8").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        NDArray point5 =
                pre.get(":, 8:10").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        return NDArrays.concat(new NDList(point1, point2, point3, point4, point5), 1);
    }

//    // decode face landmarks, 68 points per face
//    private NDArray decodeLandm68(NDArray pre, NDArray priors, double scaleXY) {
//        NDArray point1 =
//                pre.get(":, :2").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
//        NDArray point2 =
//                pre.get(":, 2:4").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
//        NDArray point3 =
//                pre.get(":, 4:6").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
//        NDArray point4 =
//                pre.get(":, 6:8").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
//        NDArray point5 =
//                pre.get(":, 8:10").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
//
//        System.out.println(pre.getShape());
//
//        NDArray point6 =
//                pre.get(":, 10:12").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
//        NDArray point7 =
//                pre.get(":, 12:14").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
//        NDArray point8 =
//                pre.get(":, 14:16").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
//
//        return NDArrays.concat(new NDList(point1, point2, point3, point4, point5, point6, point7, point8), 1);
//    }
}