package ie.atu.sw.trainer;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import static java.util.concurrent.ThreadLocalRandom.current;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class NeuralNetworkAutopilot implements IAutopilotController {
	private BasicNetwork network;
	private final int inputSize;
	private int lastMovement;

	public NeuralNetworkAutopilot(int inputSize) {
		this.inputSize = inputSize;
		createNetwork();
	}

	private void createNetwork() {
		network = new BasicNetwork();
		// Input layer: number of neurons equals the feature vector length.
		network.addLayer(new BasicLayer(null, true, inputSize));
		// Hidden layer: using a sigmoid activation; size is flexible (e.g. half of
		// input size).
		int hiddenSize = Math.max(1, inputSize / 2);
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hiddenSize));
		// Output layer: 3 neurons corresponding to movements: up, no movement, down.
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 3));
		network.getStructure().finalizeStructure();
		network.reset();
	}

	@Override
	public int getMovement(double[] state) {
		// Compute network output.
		MLData inputData = new BasicMLData(state);
		MLData output = network.compute(inputData);

		// Choose the move corresponding to the highest output neuron.
		int bestIndex = 0;
		for (int i = 1; i < output.size(); i++) {
			if (output.getData(i) > output.getData(bestIndex)) {
				bestIndex = i;
			}
		}
		// Map neuron index to movement: index 0 -> -1 (up), 1 -> 0 (straight), 2 -> 1
		// (down).
		int m = bestIndex == 0 ? -1 : bestIndex == 1 ? 0 : 1;
		System.out.println(m);
		return m;
	}

	@Override
	public void trainNetwork(List<TrainingSample> trainingData, int epochs) {
	    // Load data from CSV if no training data is provided.
	    if (trainingData.isEmpty()) {
	        try (BufferedReader br = new BufferedReader(new FileReader("training_data.csv"))) {
	            String line;
	            while ((line = br.readLine()) != null) {
	                String[] parts = line.split(",");
	                double[] features = new double[parts.length - 1];
	                for (int i = 0; i < parts.length - 1; i++) {
	                    features[i] = Double.parseDouble(parts[i]);
	                }
	                double label = Double.parseDouble(parts[parts.length - 1]);
	                trainingData.add(new TrainingSample(features, label));
	            }
	        } catch (IOException e) {
	            e.printStackTrace();
	            return;
	        }
	    }

	    // Build the complete training dataset.
	    int sampleCount = trainingData.size();
	    double[][] input = new double[sampleCount][];
	    double[][] ideal = new double[sampleCount][3]; // 3 output neurons

	    for (int i = 0; i < sampleCount; i++) {
	        TrainingSample sample = trainingData.get(i);
	        double[] feats = sample.getFeatures();
	        input[i] = feats;

	        // Determine output action.
	        // If terminal state (e.g., last feature > 0.5), force safe action (index 1).
	        boolean terminal = feats[feats.length - 1] > 0.5;
	        int outputIndex;
	        if (terminal) {
	            outputIndex = 1;
	        } else {
	            int label = (int) sample.getLabel();
	            outputIndex = label == -1 ? 0 : label == 1 ? 2 : 1;
	        }
	        ideal[i] = new double[]{0, 0, 0};
	        ideal[i][outputIndex] = 1;
	    }

	    MLDataSet trainingSet = new BasicMLDataSet(input, ideal);
	    MLTrain trainer = new ResilientPropagation(network, trainingSet);

	    // Train over the entire dataset for the given number of epochs.
	    for (int epoch = 0; epoch < epochs; epoch++) {
	        trainer.iteration();
	        System.out.println("Epoch " + epoch + " Error: " + trainer.getError());
	    }
	    trainer.finishTraining();

	    // Print final weights for debugging.
	    System.out.println("Final Weights: " + Arrays.toString(network.getFlat().getWeights()));
	}



}