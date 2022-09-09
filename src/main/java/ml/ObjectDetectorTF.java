//// Use this package name when debugging from IntelliJ
//package main.java.ml.model.ObjectDetector;
//
//// Use this package name when building with gradle to release the library
////package ml.model.ObjectDetector;
//
//import org.tensorflow.ndarray.IntNdArray;
//import org.tensorflow.ndarray.NdArrays;
//import org.tensorflow.ndarray.Shape;
//import org.tensorflow.proto.framework.DataType;
//import org.tensorflow.proto.framework.SignatureDef;
//import org.tensorflow.types.TUint8;
//
//import processing.core.*;
//
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.Map;
//import java.util.TreeMap;
//
//import static java.util.List.of;
//import static org.tensorflow.proto.framework.DataType.DT_UINT8;
//
//public class ObjectDetector {
//    SavedModelBundle model;
//    SignatureDef sig;
//    String usrPath = "/Users/jlee/src/ml4processing/";
//    String modelPath;
//    String labelPath;
//
//    private final static String[] cocoLabels = new String[] {
//            "person",
//            "bicycle",
//            "car",
//            "motorcycle",
//            "airplane",
//            "bus",
//            "train",
//            "truck",
//            "boat",
//            "traffic light",
//            "fire hydrant",
//            "street sign",
//            "stop sign",
//            "parking meter",
//            "bench",
//            "bird",
//            "cat",
//            "dog",
//            "horse",
//            "sheep",
//            "cow",
//            "elephant",
//            "bear",
//            "zebra",
//            "giraffe",
//            "hat",
//            "backpack",
//            "umbrella",
//            "shoe",
//            "eye glasses",
//            "handbag",
//            "tie",
//            "suitcase",
//            "frisbee",
//            "skis",
//            "snowboard",
//            "sports ball",
//            "kite",
//            "baseball bat",
//            "baseball glove",
//            "skateboard",
//            "surfboard",
//            "tennis racket",
//            "bottle",
//            "plate",
//            "wine glass",
//            "cup",
//            "fork",
//            "knife",
//            "spoon",
//            "bowl",
//            "banana",
//            "apple",
//            "sandwich",
//            "orange",
//            "broccoli",
//            "carrot",
//            "hot dog",
//            "pizza",
//            "donut",
//            "cake",
//            "chair",
//            "couch",
//            "potted plant",
//            "bed",
//            "mirror",
//            "dining table",
//            "window",
//            "desk",
//            "toilet",
//            "door",
//            "tv",
//            "laptop",
//            "mouse",
//            "remote",
//            "keyboard",
//            "cell phone",
//            "microwave",
//            "oven",
//            "toaster",
//            "sink",
//            "refrigerator",
//            "blender",
//            "book",
//            "clock",
//            "vase",
//            "scissors",
//            "teddy bear",
//            "hair drier",
//            "toothbrush",
//            "hair brush"
//    };
//
//    // modelNameOrUrl: A String value of a valid model OR a url to a model.json that contains a pre-trained model.
//    // Models available are: 'ssd_resnet', 'yolo'
//    public ObjectDetector(String modelName) {
//        if (modelName == "ssd_resnet") {
////            URL modelURL = getClassLoader().getResource("http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz");
////            String model = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz";
////            String labels = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt";
//
////            // You can cache the TF model on the local file system to improve the bootstrap performance on consecutive runs!
////            boolean CACHE_TF_MODEL = true;
////
////            // For the pre-trained models with mask you can set the INSTANCE_SEGMENTATION to enable object instance segmentation as well
////            boolean NO_INSTANCE_SEGMENTATION = false;
////
////            // Only object with confidence above the threshold are returned
////            float CONFIDENCE_THRESHOLD = 0.4f;
//            modelPath = usrPath + "models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model";
//
//        }
//
//        model = SavedModelBundle.load(modelPath, "serve");
//
//        // create a map of the COCO 2017 labels
//        TreeMap<Float, String> cocoTreeMap = new TreeMap<>();
//        float cocoCount = 0;
//        for (String cocoLabel : cocoLabels) {
//            cocoTreeMap.put(cocoCount, cocoLabel);
//            cocoCount++;
//        }
//
//        System.out.println("model loaded!");
//    }
//
//    public void detect(PImage image) {
//        // resize image to appropriate size
//        image.resize(640, 640);
//
//        // convert PImage to input tensors
//        // initialize input matrix with appropriate shape
//        IntNdArray inputMatrix = NdArrays.ofInts(Shape.of(1, image.width, image.height, 3));
//        // load each pixel value into input matrix
//        int width = image.width;
//        int height = image.height;
//        image.loadPixels();
//        for (int w = 0; w < width; w++) {
//            for (int h = 0; h < height; h++) {
//                inputMatrix.set(NdArrays.vectorOf(image.pixels[h*width+w] >> 16 & 0xFF), 0, w, h, 0) // r
//                        .set(NdArrays.vectorOf(image.pixels[h*width+w] >> 8 & 0xFF), 0, w, h, 1) // g
//                        .set(NdArrays.vectorOf(image.pixels[h*width+w] & 0xFF), 0, w, h, 2); // b
//            }
//        }
//
//////        Tensor<DT_UINT8> inputTensor = new Tensor<DT_UINT8>.of(DT_UINT8, inputMatrix.shape());
//////        System.out.println(inputTensor.shape());
////
//////        List<DetectedObj> result = new ArrayList<>();
////
////        Map<String, Tensor> feed_dict = new HashMap<>();
////        feed_dict.put("context", inputTensor);
//
//        // run model with the input
////        System.out.println(model.function("predict").call(feed_dict));
//    }
//}
