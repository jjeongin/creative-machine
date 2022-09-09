package ml;

public class MLLabel {
    private String label;
    private float confidence;
    public MLLabel(String label, float confidence) {
        this.label = label;
        this.confidence = confidence; // confidence score of the classification (form 0 to 1)
    }
    public String getLabel() {
        return this.label;
    }
    public float getConfidence() {
        return this.confidence;
    }
}
