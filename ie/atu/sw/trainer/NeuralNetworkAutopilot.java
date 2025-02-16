package ie.atu.sw.trainer;

import java.util.List;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.strategy.RequiredImprovementStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class NeuralNetworkAutopilot implements IAutopilotController {

	private final BasicNetwork network;

	public NeuralNetworkAutopilot(int inputSize) {
		// Build a network with a single hidden layer using Encog:
		network = new BasicNetwork();

		// Input layer:
		network.addLayer(new BasicLayer(null, true, inputSize));

		// Hidden layer (tunable size):
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 20));

		// Output layer: three neurons -> up, straight, down
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 3));

		network.getStructure().finalizeStructure();
		network.reset();
	}

	@Override
	public int getMovement(double[] state) {
		double[] output = new double[3];
		network.compute(state, output);

		// Pick the index of the highest output neuron
		int bestIndex = 0;
		double bestValue = output[0];
		for (int i = 1; i < output.length; i++) {
			if (output[i] > bestValue) {
				bestValue = output[i];
				bestIndex = i;
			}
		}
		// -1 -> up, 0 -> straight, 1 -> down
		return bestIndex == 0 ? -1 : bestIndex == 1 ? 0 : 1;
	}

	/**
	 * Train the underlying neural network with a list of TrainingSample objects.
	 * 
	 * @param samples   a list of TrainingSample (features + label)
	 * @param maxEpochs how many epochs of training to run
	 */
	@Override
	public void trainNetwork(List<TrainingSample> samples, int maxEpochs) {
		// 1) Convert your list of TrainingSample objects into an MLDataSet
		MLDataSet trainingSet = new BasicMLDataSet();
		for (TrainingSample sample : samples) {
			// Convert input features
			BasicMLData input = new BasicMLData(sample.getFeatures());
			// Convert label into a one-hot encoded array, e.g. -1 -> [1,0,0], 0 -> [0,1,0],
			// 1 -> [0,0,1]
			BasicMLData ideal = new BasicMLData(encodeLabel(sample.getLabel()));
			trainingSet.add(new BasicMLDataPair(input, ideal));
		}

		// 2) Create a ResilientPropagation trainer (other options: Backprop, etc.)
		ResilientPropagation trainer = new ResilientPropagation(network, trainingSet);

		// Optionally add a strategy to stop if no improvement after N epochs
		trainer.addStrategy(new RequiredImprovementStrategy(5));

		// 3) Run the training loop
		int epoch = 0;
		while (epoch < maxEpochs) {
			trainer.iteration();
			epoch++;
			// Optionally print out error each epoch for debugging
			// System.out.println("Epoch #" + epoch + " Error: " + trainer.getError());

			// You could add an early-stopping condition here if error drops below a
			// threshold
			if (trainer.getError() < 0.001) {
				break;
			}
		}
		trainer.finishTraining();
	}

	/**
	 * Helper function to convert the label (-1, 0, or +1) into a one-hot vector [x,
	 * y, z].
	 */
	private double[] encodeLabel(double label) {
		// label: -1 => up => [1, 0, 0]
		// label: 0 => straight => [0, 1, 0]
		// label: +1 => down => [0, 0, 1]
		double[] encoded = new double[3];
		if (label == -1) {
			encoded[0] = 1.0;
		} else if (label == 0) {
			encoded[1] = 1.0;
		} else if (label == 1) {
			encoded[2] = 1.0;
		}
		return encoded;
	}

}
