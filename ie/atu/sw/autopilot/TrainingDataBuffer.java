package ie.atu.sw.autopilot;

import java.util.ArrayList;
import java.util.List;

/**
 * A data‐oriented container that aggregates training examples in contiguous lists.
 * This allows conversion into dense primitive arrays before training, which is
 * more cache friendly.
 */
public class TrainingDataBuffer {
	private final List<double[]> featureList = new ArrayList<>();
	private final List<double[]> labelList = new ArrayList<>();

	/**
	 * Adds a new training sample.
	 * 
	 * @param features the input feature vector.
	 * @param label    the one-hot encoded label vector.
	 */
	public void addSample(double[] features, double[] label) {
		featureList.add(features);
		labelList.add(label);
	}

	/**
	 * Returns the features as a 2D array.
	 */
	public double[][] getFeaturesArray() {
		return featureList.toArray(new double[featureList.size()][]);
	}

	/**
	 * Returns the labels as a 2D array.
	 */
	public double[][] getLabelArray() {
		return labelList.toArray(new double[labelList.size()][]);
	}

	/**
	 * The number of samples stored.
	 */
	public int size() {
		return featureList.size();
	}

	/**
	 * Clears all stored samples.
	 */
	public void clear() {
		featureList.clear();
		labelList.clear();
	}
}
