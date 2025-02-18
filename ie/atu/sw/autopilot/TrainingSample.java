package ie.atu.sw.autopilot;

/**
 * A simple, immutable data carrier for a training example.
 * It is deliberately minimal to support a data‐oriented style.
 */
public class TrainingSample {
	private final double[] features;
	private final double label; // -1, 0, or 1 representing the movement

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
