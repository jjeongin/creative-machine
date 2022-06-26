# ML 4 Processing

TensorFlow ML library for [Processing](https://processing.org/).

## Build
- First, install [Adoptium OpenJDK 17](https://adoptium.net/) (required by Processing 4+)

Run gradle to build a new release package under `/release/ml4processing.zip`:

```bash
# windows
gradlew.bat releaseProcessingLib

# mac / unix
./gradlew releaseProcessingLib
```

## Developing in IntelliJ IDEA

The library can be imported as an IntelliJ project following the steps below:

- Download and install [IntelliJ IDEA](https://www.jetbrains.com/idea/download/)s
- Clone this repo and build the library following the instructions in the previous section
- Clone the [processing4 repo](https://github.com/processing/processing4)
- Create new project in IntelliJ with the name and location of your choice, e.g. ```ml-dev```
- Create new module in the project for core Processing, using as content root and the module file location the ```core``` folder under the processing4 repo. As "JARs or Directory" dependency, add ```<path to processing4 repo>/core/library```
- Create another module in the project, this time for ml4processing. Use the ml4processing's root folder as the content root and module file location. Add the processing-core module as module dependency for this module, and the ```libs``` subdirecotry inside the ml4processing directory (it should have been created during library building step) as itss "JARs or Directory" dependency
- Add the proccessing-core and ml4processing modules as dependencies in the main module of the project (ml-dev)
- You can now create a test program in under the main module of the project, for example the following code will apply a pre-generated object detection model on an input image:

```
import processing.core.*;
import ml.ObjectDetectorDJL;

public class DetectTest extends PApplet {
    ObjectDetectorDJL detector;
    PImage outputImg;

    public void settings() {
        size(parseInt(args[0]), parseInt(args[1]));
    }

    public void setup() {
        detector = new ObjectDetectorDJL(this, "mobilenet_v2");
        PImage inputImage = loadImage("dog_bike_car.jpeg");
        String detected = detector.detect(inputImage, true, "output.png");
        println("Detected objects: " + detected);
        outputImg = loadImage("output.png");
    }

    public void draw() {
        image(outputImg, 0, 0);
    }

    static public void main(String[] args) {
      PApplet.main(DetectTest.class, "768", "576");
    }
}
```

Please note that the input image should be placed inside the subdirectory ```data``` located inside the root of the IntelliJ project (i.e.: ```ml-dev/data```)