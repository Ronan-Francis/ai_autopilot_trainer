package ie.atu.sw;

public class TrainingSample {
    private final double[] features;
    private final double label; // Could be -1, 0, +1, or encoded differently

    public TrainingSample(double[] features, double label) {
        this.features = features;
        this.label = label;
    }

    public double[] getFeatures() {
        return features;
    }

    public double getLabel() {
        return label;
    }
}
