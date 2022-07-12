package ml;

public class MLKeyPoint {
    private float x, y, confidence;

    public MLKeyPoint(float x, float y, float confidence) {
        this.x = x;
        this.y = y;
        this.confidence = confidence;
    }

    public float getX() {
        return this.x;
    }

    public float getY() {
        return this.y;
    }

    public float getConfidence() {
        return this.confidence;
    }
}
