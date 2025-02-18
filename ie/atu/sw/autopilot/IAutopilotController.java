package ie.atu.sw.autopilot;

import java.util.List;

public interface IAutopilotController {
	/*
	 * Given a state vector, compute the next movement.
	 * 
	 * @param state the input state (features)
	 * @return the movement decision: -1 for up, 0 for straight, 1 for down.
	 */
	int getMovement(double[] state);

	/*
	 * Train the neural network with the given training data.
	 * 
	 * @param trainingData a list of training samples
	 * @param epochs the number of training epochs to perform
	 */
	void trainNetwork(List<TrainingSample> trainingData, int epochs);
}
