import main.java.ml.model.*;

MLTest ml;

void setup() {
  size(400,400);
  smooth();
  
  ml = new MLTest(this);
  
  PFont font = createFont("Arial", 20);
  textFont(font);
}

void draw() {
  background(0);
  fill(255);
  text("TensorFlow version " + ml.getTensorFlowVersion(), 40, 200);
}