package ie.atu.sw.autopilot;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
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
	
	/**
     * Writes the stored training data to a CSV file, including a header row.
     * The header will be "f1,f2,f3,...,label" if there are multiple features.
     *
     * @param filePath The output file path
     * @param featureCount The number of features in each sample
     */
    public void saveToCSV(String filePath, int featureCount) {
        try (PrintWriter pw = new PrintWriter(new FileWriter(filePath))) {
            // Build and write the header row (f1, f2, ... fN, label)
            StringBuilder header = new StringBuilder();
            for (int i = 1; i <= featureCount; i++) {
                header.append("f").append(i).append(",");
            }
            header.append("label");
            pw.println(header.toString());

            // Now write each sample row
            for (int i = 0; i < featureList.size(); i++) {
                double[] features = featureList.get(i);
                double[] labels = labelList.get(i); // one-hot or possibly single value

                // Print features
                StringBuilder row = new StringBuilder();
                for (double f : features) {
                    row.append(f).append(",");
                }

                // If you're storing single movement labels (e.g. -1,0,1), you’d adapt accordingly.
                // Here labelList is one-hot, so pick the index of the ‘1’ if needed.
                int movementIndex = -1; 
                for (int j = 0; j < labels.length; j++) {
                    if (labels[j] == 1.0) {
                        movementIndex = j;
                        break;
                    }
                }
                // movementIndex: 0 -> -1 (up), 1 -> 0 (straight), 2 -> 1 (down)
                int movement;
                if (movementIndex == 0) movement = -1;
                else if (movementIndex == 2) movement = 1;
                else movement = 0;

                row.append(movement);

                pw.println(row.toString());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
