package ie.atu.sw;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

public class NeuralNetworkAutopilot implements IAutopilotController {

	private final BasicNetwork network;
	
	public NeuralNetworkAutopilot(int inputSize) {
		//Build a network with a single hidden layer using Encog.
		network = new BasicNetwork();
		//Input layer: the size is determined by the feature vector(for example, a 30x20 grid).
		network.addLayer(new BasicLayer(null, true, inputSize));
		//Hidden layer: the number of neurons can be tuned as needed
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 20)); //(*)
		// Output layer: three neurons corresponding to up, straight, and down
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 3));
		network.getStructure().finalizeStructure();
		network.reset();
		
	}
	
	@Override
	public int getMovement(double[] state) {
		double [] output = new double[3];
		network.compute(state, output);
		
		// choose the output neuron with the highest value
		int bestIndex = 0;
		double bestValue = output[0];
		for (int i = 1; i < output.length; i++) {
			if(output[i] > bestValue) {
				bestValue = output[i];
				bestIndex = i;
			}
		}
		
		return bestIndex == 0 ? -1 : bestIndex == 1 ? 0 : 1;
	}

}
