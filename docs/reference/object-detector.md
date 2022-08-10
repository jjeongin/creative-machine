# Object Detector
Object Detection model 

<img src="../data/object-detector/dog_bike_car_output_from_url.png" width="500">

## Quick Start
```
// create an Object Detector
ObjectDetector detector = new ObjectDetector(this, "cocossd");

// load input image
PImage img = loadImage("data/dog_bike_car.jpeg");

// run object detection
MLObject[] output = detector.detect(img);
```

## Usage
### Initialize
```
ObjectDetector detector = new ObjectDetector(this, modelNameorURL);
```
#### Parameters
String modelNameOrURL: (required) Can be a model name of built-in models ("openimages_ssd", "cocossd", or "yolo") or a remote url/file path to a parent directory containing TensorFlow saved_model folder
### Methods
detect(PImage image): Runs object detection on [PImage](https://processing.org/reference/PImage.html). Returns an array of [MLObject]().
```
PImage img = loadImage("data/dog_bike_car.jpeg");
MLObject[] output = detector.detect(img);
```

detect(PImage image, Boolean saveBoundingBoxImage, String fileName): Runs object detection on [PImage](https://processing.org/reference/PImage.html) and a save bounding box image with the specified file name. Returns an array of [MLObject]().
```
PImage img = loadImage("data/dog_bike_car.jpeg");
MLObject[] output = detector.detect(img, true, "data/dog_bike_car_output.png");
```
## Examples
[ObjectDetectorExample](https://github.com/jjeongin/ml4processing/tree/master/examples/ObjectDetectorExample)

[ObjectDetectorFromURLExample](https://github.com/jjeongin/ml4processing/tree/master/examples/ObjectDetectorfromURLExample)
