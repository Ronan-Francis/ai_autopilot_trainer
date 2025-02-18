package ie.atu.sw.autopilot;

import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetworkAutopilot implements IAutopilotController {
	private BasicNetwork network;
	private int lastMovement = Integer.MIN_VALUE;
	private double temperature = 1.0;
	private final double minTemperature = 0.5;
	private final double maxTemperature = 2.0;

	public NeuralNetworkAutopilot(int inputSize) {
	    network = new BasicNetwork();
	    network.addLayer(new BasicLayer(null, true, inputSize));
	    int hiddenSize = Math.max(1, inputSize / 2);
	    network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hiddenSize));
	    network.addLayer(new BasicLayer(new ActivationLinear(), false, 3));
	    network.getStructure().finalizeStructure();
	    network.reset();
	}

	@Override
	public int getMovement(double[] state) {
	    MLData inputData = new BasicMLData(state);
	    MLData output = network.compute(inputData);
	    double[] activations = output.getData();

	    // Compute softmax probabilities with temperature scaling.
	    double[] probabilities = softmax(activations, temperature);

	    // Sample an index based on the computed probabilities.
	    int chosenIndex = sampleFromDistribution(probabilities);
	    int chosenMovement = movementForIndex(chosenIndex);

	    // Adjust temperature: if the same movement is repeated, increase temperature to encourage exploration.
	    if (chosenMovement == lastMovement) {
	        temperature = Math.min(maxTemperature, temperature + 0.1);
	    } else {
	        temperature = Math.max(minTemperature, temperature - 0.1);
	    }
	    lastMovement = chosenMovement;

	    //System.out.println("Chosen Movement: " + chosenMovement + " | Temperature: " + temperature);
	    return chosenMovement;
	}

	// Softmax function with temperature scaling.
	private double[] softmax(double[] activations, double temperature) {
	    double[] expValues = new double[activations.length];
	    double sum = 0;
	    for (int i = 0; i < activations.length; i++) {
	        expValues[i] = Math.exp(activations[i] / temperature);
	        sum += expValues[i];
	    }
	    for (int i = 0; i < expValues.length; i++) {
	        expValues[i] /= sum;
	    }
	    return expValues;
	}

	// Sample an index from the probability distribution.
	private int sampleFromDistribution(double[] probabilities) {
	    double rand = Math.random();
	    double cumulative = 0;
	    for (int i = 0; i < probabilities.length; i++) {
	        cumulative += probabilities[i];
	        if (rand < cumulative) {
	            return i;
	        }
	    }
	    // Fallback: return last index if rounding errors occur.
	    return probabilities.length - 1;
	}

	// Map the index to a movement: index 0 -> -1 (up), 1 -> 0 (straight), 2 -> 1 (down).
	private int movementForIndex(int index) {
	    switch (index) {
	        case 0:
	            return -1;
	        case 1:
	            return 0;
	        case 2:
	        default:
	            return 1;
	    }
	}

	@Override
	public void trainNetwork(List<TrainingSample> trainingData, int epochs) {
	    int sampleCount = trainingData.size();
	    double[][] input = new double[sampleCount][];
	    double[][] ideal = new double[sampleCount][3]; // 3 output neurons: up, neutral, down

	    for (int i = 0; i < sampleCount; i++) {
	        TrainingSample sample = trainingData.get(i);
	        double[] features = sample.getFeatures();
	        input[i] = features;

	        // Determine if this sample represents a terminal state.
	        boolean terminal = features[features.length - 1] > 0.5;
	        int outputIndex;
	        if (terminal) {
	            // Label terminal states as neutral action.
	            outputIndex = 1;
	        } else {
	            // Otherwise, determine label based on the sample's label.
	            int label = (int) sample.getLabel();
	            if (label == -1) {
	                outputIndex = 0; // up
	            } else if (label == 1) {
	                outputIndex = 2; // down
	            } else {
	                outputIndex = 1; // no movement (neutral)
	            }
	        }

	        // Create a one-hot encoded vector.
	        ideal[i] = new double[3];
	        ideal[i][outputIndex] = 1;
	    }

	    MLDataSet trainingSet = new BasicMLDataSet(input, ideal);
	    Trainer trainer = new Trainer(network, trainingSet, epochs);
	    trainer.train();
	    System.out.println("Final Weights: " + Arrays.toString(network.getFlat().getWeights()));
	}

	public static void saveTrainingSamplesToCSV(String filePath, List<TrainingSample> samples, int featureCount) {
	    try (PrintWriter pw = new PrintWriter(new FileWriter(filePath))) {
	        // header
	        StringBuilder header = new StringBuilder();
	        for (int i = 1; i <= featureCount; i++) {
	            header.append("f").append(i).append(",");
	        }
	        header.append("label");
	        pw.println(header);

	        // rows
	        for (TrainingSample ts : samples) {
	            StringBuilder row = new StringBuilder();
	            double[] features = ts.getFeatures();

	            for (double f : features) {
	                row.append(f).append(",");
	            }

	            // If ts.getLabel() is -1, 0, or 1, just append it directly.
	            row.append((int)ts.getLabel());
	            pw.println(row);
	        }
	    } catch (IOException e) {
	        e.printStackTrace();
	    }
	}

}
